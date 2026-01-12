import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version (Torch build): {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Why not available?")
    # Try to force init to see helpful errors
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"Init Error: {e}")
