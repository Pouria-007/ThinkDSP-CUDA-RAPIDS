"""Performance benchmarks for CPU vs GPU."""

import time
import numpy as np
import pytest
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

try:
    import thinkdsp
    from thinkdsp_gpu.backend import set_backend, get_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    pytestmark = pytest.mark.skip("GPU backend not available")


def benchmark_fft(size=10000, n_iter=10):
    """Benchmark FFT performance."""
    ys = np.random.randn(size).astype(np.float32)
    framerate = 11025
    
    # CPU benchmark
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    for _ in range(n_iter):
        spectrum_cpu = wave_cpu.make_spectrum()
    cpu_time = (time.time() - start) / n_iter
    
    # GPU benchmark (if available)
    backend = get_backend()
    gpu_time = None
    if backend.use_gpu:
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.make_spectrum()
        
        start = time.time()
        for _ in range(n_iter):
            spectrum_gpu = wave_gpu.make_spectrum()
        gpu_time = (time.time() - start) / n_iter
    
    set_backend(use_gpu=False)
    
    return {
        'size': size,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time if gpu_time else None
    }


def benchmark_convolution(size=10000, kernel_size=100, n_iter=10):
    """Benchmark convolution performance."""
    ys = np.random.randn(size).astype(np.float32)
    window = np.random.randn(kernel_size).astype(np.float32)
    framerate = 11025
    
    # CPU benchmark
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    for _ in range(n_iter):
        _ = wave_cpu.convolve(window)
    cpu_time = (time.time() - start) / n_iter
    
    # GPU benchmark
    backend = get_backend()
    gpu_time = None
    if backend.use_gpu:
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.convolve(window)
        
        start = time.time()
        for _ in range(n_iter):
            _ = wave_gpu.convolve(window)
        gpu_time = (time.time() - start) / n_iter
    
    set_backend(use_gpu=False)
    
    return {
        'size': size,
        'kernel_size': kernel_size,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time if gpu_time else None
    }


def benchmark_spectrogram(size=50000, seg_length=1024, n_iter=3):
    """Benchmark spectrogram performance."""
    ys = np.random.randn(size).astype(np.float32)
    framerate = 11025
    
    # CPU benchmark
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    for _ in range(n_iter):
        _ = wave_cpu.make_spectrogram(seg_length=seg_length)
    cpu_time = (time.time() - start) / n_iter
    
    # GPU benchmark
    backend = get_backend()
    gpu_time = None
    if backend.use_gpu:
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.make_spectrogram(seg_length=seg_length)
        
        start = time.time()
        for _ in range(n_iter):
            _ = wave_gpu.make_spectrogram(seg_length=seg_length)
        gpu_time = (time.time() - start) / n_iter
    
    set_backend(use_gpu=False)
    
    return {
        'size': size,
        'seg_length': seg_length,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time if gpu_time else None
    }


@pytest.mark.benchmark
def test_benchmark_fft():
    """Run FFT benchmark."""
    result = benchmark_fft(size=10000, n_iter=10)
    print(f"\nFFT Benchmark (size={result['size']}):")
    print(f"  CPU: {result['cpu_time']*1000:.2f} ms")
    if result['gpu_time']:
        print(f"  GPU: {result['gpu_time']*1000:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
    assert result['cpu_time'] > 0


@pytest.mark.benchmark
def test_benchmark_convolution():
    """Run convolution benchmark."""
    result = benchmark_convolution(size=10000, kernel_size=100, n_iter=10)
    print(f"\nConvolution Benchmark (size={result['size']}, kernel={result['kernel_size']}):")
    print(f"  CPU: {result['cpu_time']*1000:.2f} ms")
    if result['gpu_time']:
        print(f"  GPU: {result['gpu_time']*1000:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
    assert result['cpu_time'] > 0


@pytest.mark.benchmark
def test_benchmark_spectrogram():
    """Run spectrogram benchmark."""
    result = benchmark_spectrogram(size=50000, seg_length=1024, n_iter=3)
    print(f"\nSpectrogram Benchmark (size={result['size']}, seg_length={result['seg_length']}):")
    print(f"  CPU: {result['cpu_time']*1000:.2f} ms")
    if result['gpu_time']:
        print(f"  GPU: {result['gpu_time']*1000:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
    assert result['cpu_time'] > 0

