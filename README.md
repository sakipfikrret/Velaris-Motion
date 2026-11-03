# Velaris Motion: Next-Gen AI Video Generation Runtime

[![Release](https://img.shields.io/badge/Release-v2.0--Flagship-blue.svg)](https://github.com/sakipfikrret/Velaris-Motion)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20%7C%20CUDA%2012.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(34%2F34)-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-3D%20Diffusion%20Transformer%20(DiT)-purple.svg)](#-architecture--supported-engines)

**Velaris Motion** is a production-grade, hardened local video generation runtime and studio web application optimized for consumer GPUs (8GB VRAM such as NVIDIA RTX 3060 / 4060).

It integrates state-of-the-art **3D Diffusion Transformers (Wan2.1, LTX-Video, CogVideoX)**, fast **AnimateDiff-Lightning** distillation, neural super-resolution upscaling (Real-ESRGAN), cinema-grade frame interpolation (RIFE), and an autonomous **AI Director** that scripts and stitches multi-scene long-form films.

---

## 🎬 Supported Engines & Generation Modes

| Engine | Architecture | Target Output | Best For | Typical Speed (8GB VRAM) |
|---|---|---|---|---|
| 💎 **Wan2.1 (Alibaba)** *(Flagship)* | 3D DiT + UMT5-XXL + Flow Matching | 832×480 @ 16fps (81 frames, 5s) | SOTA Photorealism, physical fluid motion, zero flicker | ~33s / step |
| ⚡ **AnimateDiff-Lightning** | 4-step Distilled Motion Adapter | 512×512 (16–32 frames) | Ultra-fast preview, Director long-form films, anime/stylized | ~15–25s / shot |
| 🚀 **LTX-Video (Lightricks)** | 2B 3D DiT | 768×512 @ 24fps native | Real-time cinematic 24fps playback | ~20–35s / shot |
| 🎥 **CogVideoX-2B (THUDM)** | 3D DiT + Temporal Cross-Attention | 720×480 (49 frames, 6s) | Natural scene dynamics, semantic prompt fidelity | ~45–60s / shot |
| 📸 **Studio Photorealism** | Two-Stage (epiCRealism/SDXL + SVD-XT) | 1024×576 (14–25 frames) | DSLR-quality portrait & macro master frames | ~1–2 min / shot |
| ☁️ **Cloud Studio** | Google Veo 2 / Kling / Wan-14B API | 1080p / 4K @ 24fps | Cloud-accelerated cinematic rendering | Instant submit |

---

## 🚀 Key Features

*   **Hybrid RAM-VRAM Group & Model CPU Offloading**: Dynamically orchestrates memory between system RAM and 8GB GPU VRAM, allowing massive 1.3B–14B parameter models to run locally without Out-Of-Memory (OOM) errors.
*   **AI Director & Storyboard Mode**: Automatically decomposes narrative prompts into sequential cinematic shots, plans camera movements (aerial, dolly, pan, low-angle), generates each scene sequentially, and seamlessly stitches them into a cohesive film.
*   **Full HD (1080p) & 4K Super-Resolution**: Built-in neural upscaler (Real-ESRGAN Compact SRVGGNet) and Lanczos-3 filtering elevate raw generation to razor-sharp cinema resolutions.
*   **Smooth Motion Interpolation (RIFE / FFmpeg)**: Interpolates lower-framerate generation up to buttery-smooth 24 FPS or 60 FPS without ghosting.
*   **Hardened VRAM Orchestrator**: Strict single-active-model state machine (`UNLOADED` ➔ `LOADING` ➔ `LOADED` ➔ `INFERRING` ➔ `UNLOADING`), multi-pass garbage collection, and CUDA IPC cache management.
*   **Deterministic Reproducibility**: Emits comprehensive `{job_id}_metadata.json` audit files capturing exact seeds, model commit pins, scheduler configurations, peak VRAM, and execution timestamps.
*   **Security & Path Traversal Guards**: Strict boundary containment checks on outputs, input sanitization, and bounded parameter validation.

---

## 🛠️ System Architecture

```
                       ┌───────────────────────────────┐
                       │   Client UI (Next.js / React) │
                       └───────────────┬───────────────┘
                                       │ HTTP / REST
                                       ▼
                       ┌───────────────────────────────┐
                       │        FastAPI Gateway        │
                       │ (Input Validation & Traversal)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Serial GPU Job Queue      │
                       │   (FIFO: Deterministic GPU)   │
                       └───────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│ Wan2.1 DiT    │              │ LTX-Video DiT │              │ AnimateDiff   │
│ 1.3B / 14B    │              │ 2B @ 24fps    │              │ 4-Step LCM    │
└───────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        └──────────────────────┬───────┴──────────────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │    VRAM Memory Orchestrator   │
               │  (Group Offload / IPC Purge)  │
               └───────────────┬───────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ Post-Processing & Export Pipeline           │
        │ • RIFE Frame Interpolation (16 ➔ 24/60 fps) │
        │ • Real-ESRGAN AI Super-Resolution (➔ 1080p) │
        │ • Audio Synthesis & FFmpeg Muxing           │
        └─────────────────────────────────────────────┘
```

---

## 📦 Installation & Setup

### 1. Prerequisites
*   **Operating System**: Windows 10 / 11 (AMD64) or Linux
*   **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060, RTX 4070, or higher)
*   **CUDA**: CUDA 12.1+ / PyTorch 2.1+
*   **Python**: Python 3.10 to 3.12
*   **Node.js**: Node.js 18+ and npm

### 2. Clone & Environment Setup
```bash
git clone https://github.com/sakipfikrret/Velaris-Motion.git
cd Velaris-Motion
```

Install backend dependencies:
```bash
python -m pip install -r backend/requirements.txt
# Ensure ftfy and imageio-ffmpeg are installed
python -m pip install ftfy imageio-ffmpeg
```

If you need to install or update GPU PyTorch for CUDA 12.1:
```bash
force_gpu_install.bat
```

### 3. Launching Velaris Motion
*   **Full Stack Launcher**:
    ```bash
    run_velaris.bat
    ```
    Starts the backend AI engine at `http://127.0.0.1:8000` and opens the web application at `http://localhost:3000`.

*   **Website Only**:
    ```bash
    run_website.bat
    ```

---

## 📊 Standardized Benchmark Suite

Velaris Motion includes an end-to-end benchmark suite testing motion stability, surface tension, physical dynamics, and human anatomy:

```bash
# Run Basit (Basic) Benchmark: Swan gliding on calm mountain lake
python backend/run_benchmark_tests.py basit

# Run Orta (Medium) Benchmark: Norwegian sunset fjords drone sweep
python backend/run_benchmark_tests.py orta

# Run Zor (Hard) Benchmark: 1080p Rainy neon portrait with micro-textures
python backend/run_benchmark_tests.py zor
```

### Measured Real Hardware Results (RTX 8GB VRAM)

| Benchmark Level | Model / Engine | Resolution | Steps / Frames | Peak VRAM | Result |
|---|---|---|---|---|---|
| **Basit (Basic)** | Wan2.1 1.3B DiT | 832×480 | 30 steps / 81 frames | 5.67 GB | ✅ 100% Success (Zero flicker) |
| **Zor (Hard)** | Wan2.1 1.3B + AI Upscaler | **1920×1080 FHD** | 30 steps / 81 frames (24fps) | 5.70 GB | ✅ 100% Success (Pristine detail) |

---

## 🧪 Automated Unit & Integration Tests

The test suite covers model registration, configuration integrity, boundary validation, security path containment, frame interpolation, and DiT engine dispatch:

```bash
# Run complete test suite (34 passing tests)
$env:VELARIS_MOCK_MODE="1"; python -m pytest tests/ -v
```

---

## 🔒 Security & Local Hardening

*   **Path Traversal Prevention**: File retrieval endpoints strictly enforce that resolved paths reside inside the designated `outputs` folder using boundary containment checks.
*   **Filename Sanitization**: Rejects paths containing `..`, traversal slashes, null bytes, or unauthorized file extensions.
*   **Resource Bounds**: Strict limits on prompt lengths (1–1000 characters), inference steps, and concurrent tasks.
*   **Isolated Workspace**: Zero external network telemetry required for local inference.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
