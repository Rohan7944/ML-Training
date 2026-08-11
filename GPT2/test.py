import sys
import torch

print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version:   {torch.__version__}")
print(f"CUDA Available:    {torch.cuda.is_available()}")