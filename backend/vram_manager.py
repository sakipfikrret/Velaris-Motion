"""
Velaris Motion - VRAM Orchestrator & Hardware State Machine
Target Release: v1.1.0
"""

import gc
import sys
import threading
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import torch

logger = logging.getLogger("velaris.vram")


class VRAMState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INFERRING = "inferring"
    UNLOADING = "unloading"
    ERROR = "error"


class VRAMManager:
    """
    Orchestrates GPU memory allocations, releases, and state transitions.
    Guarantees that consumer GPUs (such as 8GB RTX 4060) never have two
    large models co-resident in VRAM.
    """
    def __init__(self, device: Optional[str] = None):
        self._lock = threading.RLock()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._state = VRAMState.UNLOADED
        self._current_model: Optional[str] = None
        self._active_pipeline_ref: Optional[Any] = None
        self._peak_vram_mb: float = 0.0

    @property
    def state(self) -> VRAMState:
        with self._lock:
            return self._state

    @property
    def current_model(self) -> Optional[str]:
        with self._lock:
            return self._current_model

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns comprehensive hardware VRAM and allocation metrics."""
        with self._lock:
            stats: Dict[str, Any] = {
                "device": self.device,
                "state": self._state.value,
                "current_model": self._current_model,
                "cuda_available": torch.cuda.is_available(),
                "device_name": None,
                "total_vram_mb": 0.0,
                "free_vram_mb": 0.0,
                "allocated_vram_mb": 0.0,
                "reserved_vram_mb": 0.0,
                "peak_vram_mb": self._peak_vram_mb,
            }

            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    stats["device_name"] = torch.cuda.get_device_name(0)
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    stats["total_vram_mb"] = round(total_bytes / (1024 * 1024), 2)
                    stats["free_vram_mb"] = round(free_bytes / (1024 * 1024), 2)
                    
                    allocated_bytes = torch.cuda.memory_allocated()
                    reserved_bytes = torch.cuda.memory_reserved()
                    stats["allocated_vram_mb"] = round(allocated_bytes / (1024 * 1024), 2)
                    stats["reserved_vram_mb"] = round(reserved_bytes / (1024 * 1024), 2)
                    
                    max_alloc = torch.cuda.max_memory_allocated()
                    peak_now = round(max_alloc / (1024 * 1024), 2)
                    stats["peak_vram_mb"] = max(self._peak_vram_mb, peak_now)
                except Exception as e:
                    stats["error"] = str(e)
            
            return stats

    def prepare_for_model(self, model_id: str, auto_unload: bool = True):
        """
        State machine enforcement: Verifies that no other model is occupying VRAM.
        If auto_unload is True, gracefully unloads any existing model first.
        """
        with self._lock:
            if self._current_model == model_id and self._state == VRAMState.LOADED:
                return  # Model already ready

            if self._current_model is not None:
                if auto_unload:
                    self.unload_current_model()
                else:
                    raise RuntimeError(
                        f"Cannot load model '{model_id}': Model '{self._current_model}' is currently resident in VRAM. "
                        f"Current state: {self._state.value}."
                    )

            self._state = VRAMState.LOADING
            self._current_model = model_id

    def set_model_loaded(self, model_id: str, pipeline_ref: Any):
        """Confirms model loading has succeeded and registers the live reference."""
        with self._lock:
            self._current_model = model_id
            self._active_pipeline_ref = pipeline_ref
            self._state = VRAMState.LOADED
            if self.device == "cuda" and torch.cuda.is_available():
                self._update_peak()

    def apply_hybrid_ram_vram_offload(
        self,
        pipeline: Any,
        enable_group_offload: bool = True,
        num_blocks_per_group: int = 1,
        use_stream: bool = True,
    ) -> bool:
        """
        Orchestrates hybrid execution between 32GB system RAM and 8GB GPU VRAM.
        Streams transformer blocks dynamically via CUDA stream or falls back
        to standard model CPU offload.
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            if hasattr(pipeline, "to"):
                pipeline.to("cpu")
            return False

        # Attempt modern block-level group offloading (Diffusers 0.36+)
        if enable_group_offload and hasattr(pipeline, "enable_group_offload"):
            try:
                pipeline.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=num_blocks_per_group,
                    use_stream=use_stream,
                )
                logger.info(
                    f"Hybrid RAM-VRAM Group Offload enabled (num_blocks={num_blocks_per_group}, stream={use_stream})."
                )
                return True
            except Exception as err:
                logger.warning(f"Group offload not applicable ({err}), falling back to model_cpu_offload.")

        # Fallback to model-level CPU offload
        if hasattr(pipeline, "enable_model_cpu_offload"):
            try:
                pipeline.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled.")
                return True
            except Exception as err:
                logger.warning(f"Model CPU offload failed ({err}).")

        if hasattr(pipeline, "to"):
            pipeline.to("cuda")
        return False

    def unload_current_model(self):
        """
        Rigorous GPU cleanup:
        1. Set state to UNLOADING
        2. Break and delete model / pipeline references
        3. Multi-pass garbage collection (cycles)
        4. Empty CUDA cache & IPC collect
        5. Reset peak stats
        6. Transition to UNLOADED
        """
        with self._lock:
            if self._current_model is None and self._state == VRAMState.UNLOADED:
                return

            self._state = VRAMState.UNLOADING
            
            # Explicit reference cleanup
            if self._active_pipeline_ref is not None:
                try:
                    # Move components to cpu if possible to accelerate dealloc
                    if hasattr(self._active_pipeline_ref, "to"):
                        self._active_pipeline_ref.to("cpu")
                except Exception:
                    pass
                del self._active_pipeline_ref
                self._active_pipeline_ref = None

            self._current_model = None

            # Multi-pass GC to collect cyclical references in PyTorch module graphs
            for _ in range(3):
                gc.collect()

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                self._update_peak()

            self._state = VRAMState.UNLOADED

    @contextmanager
    def managed_inference(self, model_id: str):
        """
        Context manager ensuring state machine updates and automated VRAM cleanup
        on errors or OOM exceptions.
        """
        with self._lock:
            if self._current_model != model_id or self._state != VRAMState.LOADED:
                raise RuntimeError(
                    f"Model '{model_id}' is not loaded. Current model: '{self._current_model}', state: '{self._state}'"
                )
            self._state = VRAMState.INFERRING

        try:
            yield
            with self._lock:
                self._state = VRAMState.LOADED
                self._update_peak()
        except Exception as exc:
            with self._lock:
                self._state = VRAMState.ERROR
            self.safe_cleanup()
            raise exc

    def safe_cleanup(self):
        """Emergency cleanup invoked upon exceptions or cancellation."""
        with self._lock:
            try:
                self.unload_current_model()
            except Exception:
                pass
            finally:
                self._state = VRAMState.UNLOADED
                self._current_model = None
                self._active_pipeline_ref = None
                for _ in range(3):
                    gc.collect()
                if self.device == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    def _update_peak(self):
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                current_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if current_peak > self._peak_vram_mb:
                    self._peak_vram_mb = round(current_peak, 2)
            except Exception:
                pass


# Global singleton instance
vram_manager = VRAMManager()
