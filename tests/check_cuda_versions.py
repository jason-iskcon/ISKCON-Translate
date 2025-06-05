import torch
import subprocess
import sys

print("=== CUDA Version Check ===")

# Check PyTorch CUDA version
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

# Check if cuDNN is available
if torch.backends.cudnn.is_available():
    print("cuDNN version:", torch.backends.cudnn.version())
else:
    print("cuDNN: Not available")

# Try to get system CUDA version
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                print("System CUDA version:", line.split('CUDA Version:')[1].split()[0])
                break
    else:
        print("nvidia-smi failed:", result.stderr)
except Exception as e:
    print("Could not run nvidia-smi:", e)

# Check if we can create a CUDA tensor
try:
    x = torch.tensor([1.0]).cuda()
    print("✓ Basic CUDA tensor creation works")
except Exception as e:
    print("✗ CUDA tensor creation failed:", e)
    print("This likely indicates a CUDA library mismatch")

print("\n=== Diagnosis ===")
if "cu118" in torch.__version__:
    print("PyTorch is built for CUDA 11.8")
elif "cu121" in torch.__version__:
    print("PyTorch is built for CUDA 12.1")
elif "cu124" in torch.__version__:
    print("PyTorch is built for CUDA 12.4")
else:
    print("Cannot determine PyTorch CUDA version from:", torch.__version__)

print("\nIf you see 'cublas64_12.dll' errors, it means:")
print("1. PyTorch wants CUDA 12.x libraries")
print("2. But system has CUDA 11.x or incomplete CUDA 12.x installation")
print("3. Solution: Reboot first, then potentially reinstall PyTorch for correct CUDA version") 