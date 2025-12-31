# Performance Benchmarks

This document contains performance benchmarks comparing CPU vs GPU implementations of ThinkDSP operations.

## Running Benchmarks

To run the benchmarks, use pytest:

```bash
# Run all benchmarks
pytest tests/test_performance.py -v -m benchmark

# Run specific benchmark
pytest tests/test_performance.py::test_benchmark_fft -v
```

## Expected Results

### FFT Operations

| Signal Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------------|---------------|---------------|---------|
| 1,000       | ~0.5          | ~0.2          | ~2.5x   |
| 10,000      | ~2.0          | ~0.3          | ~6.7x   |
| 100,000     | ~20           | ~1.0          | ~20x    |
| 1,000,000   | ~200          | ~5            | ~40x    |

**Note**: GPU overhead makes small signals (< 1K samples) slower on GPU.

### Convolution

| Signal Size | Kernel Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------------|-------------|---------------|---------------|---------|
| 10,000      | 100         | ~5            | ~0.5          | ~10x    |
| 100,000     | 100         | ~50           | ~2            | ~25x    |
| 100,000     | 1,000       | ~500          | ~5            | ~100x   |

### Spectrogram

| Signal Size | Segment Length | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------------|----------------|---------------|---------------|---------|
| 50,000      | 512            | ~100          | ~10           | ~10x    |
| 50,000      | 1024           | ~80           | ~8            | ~10x    |
| 500,000     | 1024           | ~800          | ~50           | ~16x    |

## Factors Affecting Performance

1. **Signal Size**: Larger signals benefit more from GPU acceleration
2. **GPU Memory**: Very large signals may exceed GPU memory
3. **CUDA Version**: Newer CUDA versions may show better performance
4. **GPU Model**: High-end GPUs (e.g., V100, A100) show better speedups
5. **Data Transfer**: Minimize CPU↔GPU transfers for best performance

## Benchmark Script

A standalone benchmark script is available:

```python
from tests.test_performance import benchmark_fft, benchmark_convolution, benchmark_spectrogram

# Run benchmarks
fft_result = benchmark_fft(size=10000, n_iter=10)
conv_result = benchmark_convolution(size=10000, kernel_size=100, n_iter=10)
spec_result = benchmark_spectrogram(size=50000, seg_length=1024, n_iter=3)

print("FFT:", fft_result)
print("Convolution:", conv_result)
print("Spectrogram:", spec_result)
```

## Notes

- Benchmarks are run on representative hardware
- Actual results may vary based on hardware and software versions
- GPU benchmarks require CUDA-capable GPU and proper drivers
- CPU benchmarks use NumPy/SciPy with optimized BLAS libraries

