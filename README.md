# Velaris Motion: Local Hybrid AI Video Generator


**Velaris Motion** represents a new approach to local video generation. Instead of relying on heavy, closed-source APIs, this project implements a custom "Hybrid Architecture" specifically optimized for consumer hardware (like the RTX 4060).

It combines a high-fidelity **Dreamer** (Text-to-Image) with a stable **Animator** (Image-to-Video) to produce cinematic results that outperform standard single-stage models.

## 🚀 Features

*   **Hybrid Architecture**: Two-stage pipeline (Dreamer + Animator) for superior aesthetic quality.
*   **Local Privacy**: Runs 100% locally. No data leaves your machine.
*   **Hardware Optimized**: Custom "VRAM Orchestrator" allows running large fp16 models on 8GB VRAM cards (RTX 3060/4060).
*   **One-Click Install**: Includes automated `.bat` scripts for setup and execution.
*   **CPU Fallback**: Auto-detects hardware; runs safely on CPU if GPU is unavailable.
*   **Modern UI**: Built with Next.js 16, Tailwind CSS v4, and Glassmorphism aesthetics.

## 🛠️ Architecture

The **Velaris Engine** (`backend/engine.py`) orchestrates two specialized models:

```python
# Hybrid Pipeline Logic (Simplified)
class HybridEngine:
    def generate(self, prompt):
        # Stage 1: The Dreamer (Stable Diffusion 1.5 Fine-tune)
        # Generates the perfect initial frame based on your text.
        self.load_dreamer()
        image = self.txt2img_pipe(prompt).images[0]
        self.unload_dreamer() # Frees 4GB VRAM
        
        # Stage 2: The Animator (SVD-XT)
        # Breathes life into the static image.
        self.load_animator()
        frames = self.img2vid_pipe(image).frames[0]
        self.unload_animator()
        
        return export_to_video(frames)
```

## 📦 Installation

We have automated the entire process for Windows.

### 1. Prerequisites
*   Windows 10/11
*   NVIDIA GPU (Recommended: RTX 3060 or better)
*   Python 3.10+ installed

### 2. Setup
Clone the repo and run the installer:
```bash
git clone https://github.com/sakipfikrret/velaris-motion.git
cd velaris-motion
```

**Double-click** `run_veo.bat`.
*   This will create a virtual environment, install PyTorch (CUDA enabled), and download the models (~5GB).

### 3. Usage
1.  **Start Backend**: Double-click `run_veo.bat` (Keep the black window open).
2.  **Start Frontend**: Double-click `run_website.bat`.
3.  Go to `http://localhost:3000` and start dreaming!

## 🔧 Tech Stack

*   **Frontend**: Next.js 16, React, Tailwind CSS, Lucide Icons.
*   **Backend**: Python, FastAPI, Uvicorn.
*   **AI Models**:
    *   *Dreamer*: `Lykon/dreamshaper-8`
    *   *Animator*: `stabilityai/stable-video-diffusion-img2vid-xt`
*   **Acceleration**: PyTorch (CUDA 12.1), xFormers (optional).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by Fikret.*
