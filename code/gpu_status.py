"""GPU status utility for notebooks.

Add this cell to your notebook to check GPU status:

```python
from gpu_status import print_gpu_status
print_gpu_status()
```
"""

try:
    from thinkdsp_gpu.backend import get_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


def print_gpu_status():
    """Print GPU backend status information."""
    if not BACKEND_AVAILABLE:
        print("GPU backend not available. Install cupy and cusignal for GPU support.")
        return
    
    backend = get_backend()
    print(f"Backend: {backend.name.upper()}")
    
    if backend.use_gpu:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            mem_info = cp.cuda.Device().mem_info
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            
            print(f"✓ GPU Acceleration Enabled")
            print(f"  Device: {device.id}")
            print(f"  Name: {props['name'].decode()}")
            print(f"  Memory: {mem_info[0] / 1e9:.2f} GB free / {mem_info[1] / 1e9:.2f} GB total")
            print(f"  Compute Capability: {props['major']}.{props['minor']}")
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    else:
        print("CPU mode (NumPy/SciPy)")
        print("  To enable GPU: install cupy and cusignal")
        print("  Or set: export THINKDSP_BACKEND=gpu")


if __name__ == "__main__":
    print_gpu_status()

