import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
print(f"CUDA Version (Torch build): {cuda_build}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Why not available?")
    # Try to force init to see helpful errors
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"Init Error: {e}")
