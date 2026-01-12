import torch
import gc
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import os
import uuid

class HybridEngine:
    def __init__(self):
        # Auto-detect optimal device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            print("✅ GPU DETECTED: Running in FAST mode (CUDA + fp16)")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("⚠️ GPU NOT FOUND: Running in COMPATIBILITY mode (CPU + fp32). This will be slow.")

        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # We process models sequentially to save VRAM
        self.txt2img_pipe = None
        self.img2vid_pipe = None

    def flush(self):
        """Releases VRAM by moving models to CPU and running garbage collection."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def load_dreamer(self):
        """Stage 1: Load the Image Generator (Dreamer)."""
        if self.txt2img_pipe is not None:
            return
            
        print("LOADING DREAMER (Stage 1)...")
        # Using a high-quality SD 1.5 model for efficiency and aesthetics
        model_id = "Lykon/dreamshaper-8" 
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            use_safetensors=True
        ).to(self.device)
        
        if self.device == "cuda":
            self.txt2img_pipe.enable_model_cpu_offload()

    def unload_dreamer(self):
        if self.txt2img_pipe:
            del self.txt2img_pipe
            self.txt2img_pipe = None
            self.flush()

    def load_animator(self):
        """Stage 2: Load the Video Generator (Animator)."""
        if self.img2vid_pipe is not None:
            return
            
        print("LOADING ANIMATOR (Stage 2)...")
        model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.img2vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            variant="fp16" if self.device == "cuda" else None
        ).to(self.device)
        
        if self.device == "cuda":
            self.img2vid_pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, num_inference_steps=25, motion_bucket_id=127):
        session_id = str(uuid.uuid4())
        
        # --- STAGE 1: DREAM (Text to Image) ---
        self.load_dreamer()
        print(f"Dreaming: {prompt}")
        image = self.txt2img_pipe(
            prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save temp image for debugging
        image_path = os.path.join(self.output_dir, f"{session_id}_dream.png")
        image.save(image_path)
        image = image.resize((1024, 576)) # Resize for SVD-XT
        
        # Unload Dreamer to free VRAM for Animator
        self.unload_dreamer()
        
        # --- STAGE 2: ANIMATE (Image to Video) ---
        self.load_animator()
        print("Animating...")
        
        frames = self.img2vid_pipe(
            image, 
            decode_chunk_size=2, # Perform decoding in smaller chunks to save VRAM
            generator=torch.manual_seed(42),
            motion_bucket_id=motion_bucket_id,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        video_path = os.path.join(self.output_dir, f"{session_id}.mp4")
        export_to_video(frames, video_path, fps=7)
        
        print(f"DONE! Video saved to: {video_path}")
        return video_path
