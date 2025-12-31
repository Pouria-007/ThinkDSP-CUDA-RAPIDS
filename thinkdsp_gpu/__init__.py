"""GPU-accelerated backend for ThinkDSP.

This package provides GPU acceleration for ThinkDSP using CuPy and cuSignal
while maintaining CPU fallback compatibility.
"""

from thinkdsp_gpu.backend import (
    get_backend,
    set_backend,
    to_cpu,
    to_gpu,
    is_gpu_array,
    Backend,
)

__all__ = [
    "get_backend",
    "set_backend",
    "to_cpu",
    "to_gpu",
    "is_gpu_array",
    "Backend",
]

