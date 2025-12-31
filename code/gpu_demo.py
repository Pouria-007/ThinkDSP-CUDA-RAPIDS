"""GPU acceleration demo for ThinkDSP.

This script demonstrates GPU acceleration capabilities.
Run with: python code/gpu_demo.py
"""

import sys
import os
import time
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))

import thinkdsp
from thinkdsp_gpu.backend import get_backend, set_backend


def print_backend_info():
    """Print information about the current backend."""
    backend = get_backend()
    print(f"\n{'='*60}")
    print(f"Backend: {backend.name.upper()}")
    print(f"{'='*60}")
    
    if backend.use_gpu:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            mem_info = cp.cuda.Device().mem_info
            print(f"GPU Device: {device.id}")
            print(f"GPU Name: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
            print(f"GPU Memory: {mem_info[1] / 1e9:.2f} GB total")
            print(f"GPU Memory Free: {mem_info[0] / 1e9:.2f} GB")
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    else:
        print("Using CPU (NumPy/SciPy)")
    
    print()


def demo_fft():
    """Demonstrate FFT acceleration."""
    print("Demo 1: FFT Acceleration")
    print("-" * 60)
    
    # Create a test signal
    duration = 0.5
    framerate = 44100
    freq = 440
    n = int(duration * framerate)
    ts = np.arange(n) / framerate
    ys = np.sin(2 * np.pi * freq * ts) + 0.5 * np.sin(2 * np.pi * freq * 2 * ts)
    
    # CPU
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    spectrum_cpu = wave_cpu.make_spectrum()
    cpu_time = time.time() - start
    
    print(f"CPU FFT: {cpu_time*1000:.2f} ms")
    print(f"  Signal size: {len(ys):,} samples")
    print(f"  Max amplitude: {np.max(spectrum_cpu.amps):.4f}")
    
    # GPU (if available)
    backend = get_backend()
    if backend.use_gpu:
        set_backend(use_gpu=True)
        backend = get_backend()  # Get updated backend
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.make_spectrum()
        
        start = time.time()
        spectrum_gpu = wave_gpu.make_spectrum()
        gpu_time = time.time() - start
        
        print(f"GPU FFT: {gpu_time*1000:.2f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify correctness
        amps_cpu = spectrum_cpu.amps
        amps_gpu = backend.to_cpu(spectrum_gpu.amps)
        max_diff = np.max(np.abs(amps_cpu - amps_gpu))
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Using GPU: {backend.is_gpu_array(spectrum_gpu.hs)}")
    
    set_backend(use_gpu=False)
    print()


def demo_convolution():
    """Demonstrate convolution acceleration."""
    print("Demo 2: Convolution Acceleration")
    print("-" * 60)
    
    # Create test signal and kernel
    duration = 0.5
    framerate = 44100
    n = int(duration * framerate)
    ys = np.random.randn(n).astype(np.float32)
    window = np.hamming(100)
    
    # CPU
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    convolved_cpu = wave_cpu.convolve(window)
    cpu_time = time.time() - start
    
    print(f"CPU Convolution: {cpu_time*1000:.2f} ms")
    print(f"  Signal size: {len(ys):,} samples")
    print(f"  Kernel size: {len(window)}")
    
    # GPU (if available)
    backend = get_backend()
    if backend.use_gpu:
        set_backend(use_gpu=True)
        backend = get_backend()  # Get updated backend
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.convolve(window)
        
        start = time.time()
        convolved_gpu = wave_gpu.convolve(window)
        gpu_time = time.time() - start
        
        print(f"GPU Convolution: {gpu_time*1000:.2f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify correctness
        max_diff = np.max(np.abs(convolved_cpu.ys - convolved_gpu.ys))
        print(f"  Max difference: {max_diff:.2e}")
    
    set_backend(use_gpu=False)
    print()


def demo_spectrogram():
    """Demonstrate spectrogram acceleration."""
    print("Demo 3: Spectrogram Acceleration")
    print("-" * 60)
    
    # Create test signal
    duration = 2.0
    framerate = 44100
    n = int(duration * framerate)
    ts = np.arange(n) / framerate
    # Chirp signal
    freqs = 440 + (880 - 440) * ts / duration
    ys = np.sin(2 * np.pi * freqs * ts)
    
    # CPU
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    
    start = time.time()
    specgram_cpu = wave_cpu.make_spectrogram(seg_length=1024)
    cpu_time = time.time() - start
    
    print(f"CPU Spectrogram: {cpu_time*1000:.2f} ms")
    print(f"  Signal size: {len(ys):,} samples")
    print(f"  Segments: {len(specgram_cpu.times())}")
    
    # GPU (if available)
    backend = get_backend()
    if backend.use_gpu:
        set_backend(use_gpu=True)
        backend = get_backend()  # Get updated backend
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        
        # Warmup
        _ = wave_gpu.make_spectrogram(seg_length=1024)
        
        start = time.time()
        specgram_gpu = wave_gpu.make_spectrogram(seg_length=1024)
        gpu_time = time.time() - start
        
        print(f"GPU Spectrogram: {gpu_time*1000:.2f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    
    set_backend(use_gpu=False)
    print()


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("ThinkDSP GPU Acceleration Demo")
    print("="*60)
    
    print_backend_info()
    
    demo_fft()
    demo_convolution()
    demo_spectrogram()
    
    print("="*60)
    print("Demo complete!")
    print("="*60)
    print("\nFor more information, see:")
    print("  - docs/GPU_MIGRATION.md")
    print("  - docs/PERFORMANCE.md")
    print("  - tests/test_correctness.py")
    print("  - tests/test_performance.py")


if __name__ == "__main__":
    main()

