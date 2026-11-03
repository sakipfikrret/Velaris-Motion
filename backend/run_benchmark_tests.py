"""
Velaris Motion - Benchmark Suite (Basit, Orta, Zor)
Tests generation quality, motion stability, VRAM management, and performance across 3 difficulty levels.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, cast

# Force unbuffered output
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Disable diffusers tqdm progress bar on Windows to prevent console handle access violations
try:
    from diffusers.utils.logging import disable_progress_bar
    disable_progress_bar()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import config
from security import SafeGenerationRequest
from engine import HybridEngine
from job_queue import Job

logger = logging.getLogger("velaris.benchmark")

TEST_SUITE: Dict[str, Dict[str, Any]] = {
    "basit": {
        "title": "Basit Test: Sakin Göl ve Kuğu (Temel Hareket & Denge)",
        "prompt": "A graceful white swan gliding smoothly on a calm crystal-clear mountain lake, gentle ripple reflections, soft morning sunlight, peaceful cinematic nature",
        "engine": "wan",
        "aspect_ratio": "16:9",
        "quality": "draft",
        "expected_duration_est": "~1.5-2.5 min",
        "focus": "Hareket stabilitesi, yüzey gerilimi, artifact/kumlanma olmaması",
    },
    "orta": {
        "title": "Orta Test: Norveç Fiyordunda Gün Batımı (Akışkan Doğa & Atmosfer)",
        "prompt": "Dramatic sunset over majestic Norwegian snowy fjords, steep granite peaks glowing with golden hour light, calm dark fjord water with amber reflections, soft atmospheric haze, sweeping cinematic drone motion, 35mm film",
        "engine": "wan",
        "aspect_ratio": "16:9",
        "quality": "hd",
        "expected_duration_est": "~2-4 min",
        "focus": "Işık kırılmaları, 3D uzamsal derinlik, erime/flicker direnci",
    },
    "zor": {
        "title": "Zor Test: Yağmur Altında Gece Portresi (Mikro Doku, Karakter & Yansımalar)",
        "prompt": "Cinematic 85mm f/1.4 portrait of a young woman walking through a rainy neon-lit city street at night, wet strands of hair, raindrops falling, vibrant amber and teal reflections in asphalt puddles, shallow depth of field, authentic skin texture with visible pores, master cinematography",
        "engine": "wan",
        "aspect_ratio": "16:9",
        "quality": "fhd",
        "expected_duration_est": "~3-5 min",
        "focus": "İnsan anatomisi, ıslak saç/yüz detayları, karmaşık neon yansımaları ve stres testi",
    },
}

EngineType = Literal["auto", "wan", "ltx", "studio", "studio_photoreal", "cogvideox", "animatediff", "cloud"]
AspectRatioType = Literal["16:9", "9:16", "1:1"]
QualityType = Literal["draft", "hd", "fhd", "4k"]


def run_single_test(test_key: str, engine_override: Optional[str] = None) -> Dict[str, Any]:
    if test_key not in TEST_SUITE:
        raise ValueError(f"Unknown test: {test_key}. Available: {list(TEST_SUITE.keys())}")

    spec = TEST_SUITE[test_key]
    engine_name = engine_override or str(spec["engine"])

    def safe_print(*args: Any, **kwargs: Any) -> None:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            clean = " ".join(str(a) for a in args).encode("ascii", errors="replace").decode("ascii")
            print(clean, **kwargs)

    safe_print(f"\n{'='*70}", flush=True)
    safe_print(f"[>>] BASLIYOR: [{test_key.upper()}] - {spec['title']}", flush=True)
    safe_print(f"Engine: {engine_name} | Kalite: {spec['quality']} | Odak: {spec['focus']}", flush=True)
    safe_print(f"Prompt: {spec['prompt']}", flush=True)
    safe_print(f"{'='*70}\n", flush=True)

    engine = HybridEngine(mock_mode=False)

    safe_engine = cast(EngineType, engine_name)
    safe_ar = cast(AspectRatioType, spec["aspect_ratio"])
    safe_quality = cast(QualityType, spec["quality"])

    req = SafeGenerationRequest(
        prompt=str(spec["prompt"]),
        engine=safe_engine,
        aspect_ratio=safe_ar,
        quality=safe_quality,
        interpolate=True,
    )
    job = Job(request=req)

    t0 = time.time()
    try:
        engine.execute_job(job)
        elapsed = round(time.time() - t0, 1)

        v_path = job.video_path or ""
        video_path = Path(v_path)
        size_mb = round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else 0

        vram_stats = engine.vram.get_memory_stats()
        peak_vram = vram_stats.get("peak_vram_mb", 0)

        result = {
            "test_level": test_key,
            "status": "SUCCESS",
            "title": spec["title"],
            "video_path": str(video_path),
            "video_size_mb": size_mb,
            "elapsed_seconds": elapsed,
            "peak_vram_mb": peak_vram,
            "metadata": job.metadata,
        }
        safe_print(f"\n[OK] [{test_key.upper()}] TAMAMLANDI!", flush=True)
        safe_print(f"Video: {video_path.name} ({size_mb} MB)", flush=True)
        safe_print(f"Sure: {elapsed} saniye | Tepe VRAM: {peak_vram} MB", flush=True)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "test_level": test_key,
            "status": "FAILED",
            "error": str(e),
            "elapsed_seconds": round(time.time() - t0, 1),
        }


if __name__ == "__main__":
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "basit"
    eng_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_single_test(test_arg, engine_override=eng_arg)
