# GPU Migration Plan for ThinkDSP

This document outlines the migration of ThinkDSP from CPU-only to GPU-accelerated computation using CUDA, CuPy, and cuSignal.

## Overview

The goal is to add GPU acceleration while maintaining:
- **API Compatibility**: Public API remains unchanged
- **CPU Fallback**: All operations work on CPU-only machines
- **Educational Value**: Code remains readable and educational

## Architecture

### Backend Abstraction Layer

A thin backend layer (`thinkdsp_gpu/backend.py`) provides:
- Automatic GPU detection and selection
- Unified interface for CPU/GPU operations
- Conversion utilities for host/device transfers

### Operation Mappings

| CPU Operation | GPU Equivalent | Notes |
|---------------|----------------|-------|
| `numpy.fft.fft` | `cupy.fft.fft` | Direct mapping |
| `numpy.fft.ifft` | `cupy.fft.ifft` | Direct mapping |
| `numpy.fft.rfft` | `cupy.fft.rfft` | Direct mapping |
| `numpy.fft.irfft` | `cupy.fft.irfft` | Direct mapping |
| `numpy.fft.fftfreq` | `cupy.fft.fftfreq` | Direct mapping |
| `numpy.fft.rfftfreq` | `cupy.fft.rfftfreq` | Direct mapping |
| `numpy.fft.fftshift` | `cupy.fft.fftshift` | Direct mapping |
| `numpy.fft.ifftshift` | `cupy.fft.ifftshift` | Direct mapping |
| `numpy.convolve` | `cusignal.convolve` or `cupy.convolve` | Prefer cuSignal for signal processing |
| `numpy.hamming` | `cusignal.windows.hamming` | Window functions via cuSignal |
| `scipy.fftpack.dct` | `cusignal.dct` or CuPy implementation | DCT via cuSignal |
| `scipy.fftpack.idct` | `cusignal.idct` or CuPy implementation | IDCT via cuSignal |
| `scipy.signal.gaussian` | `cusignal.windows.gaussian` | Gaussian window via cuSignal |
| `scipy.signal.fftconvolve` | `cusignal.fftconvolve` | FFT-based convolution |

## Key Design Decisions

### 1. Array Backend Selection

- **Default**: Auto-detect GPU if CuPy is available and CUDA device exists
- **Override**: `THINKDSP_BACKEND=cpu|gpu` environment variable
- **Fallback**: Always fall back to CPU if GPU unavailable

### 2. Memory Management

- **RAPIDS-style**: Keep arrays on GPU once moved
- **Minimize transfers**: Only convert to CPU for:
  - Matplotlib plotting (requires CPU arrays)
  - File I/O (WAV files)
  - NumPy-only operations (e.g., scipy.stats)

### 3. Data Type Handling

- Preserve original dtypes (float32/float64)
- CuPy arrays maintain dtype compatibility with NumPy
- Document any numerical precision differences

### 4. Visualization

- Convert GPU arrays to CPU only at plotting boundaries
- Use `backend.to_cpu()` utility for matplotlib operations
- Keep computation on GPU until visualization

## Implementation Phases

### Phase 1: Backend Infrastructure ✅
- [x] Create backend abstraction layer
- [x] Implement auto-detection
- [x] Add conversion utilities
- [x] Environment variable override

### Phase 2: Core Operations
- [ ] Replace FFT operations in `Wave.make_spectrum()`
- [ ] Replace FFT operations in `Spectrum.make_wave()`
- [ ] Replace convolution in `Wave.convolve()`
- [ ] Replace window functions (hamming, etc.)
- [ ] Replace DCT operations

### Phase 3: Advanced Features
- [ ] Spectrogram computation
- [ ] Filter operations (low_pass, high_pass, etc.)
- [ ] Signal generation (keep on CPU for now, or move to GPU)

### Phase 4: Testing & Validation
- [ ] Correctness tests (CPU vs GPU comparison)
- [ ] Performance benchmarks
- [ ] Notebook updates

### Phase 5: Documentation
- [ ] Update README with GPU instructions
- [ ] Add GPU status cells to notebooks
- [ ] Performance documentation

## Files Modified

### Core Library
- `code/thinkdsp.py` - Main DSP library (uses backend)
- `thinkdsp_gpu/backend.py` - Backend abstraction (NEW)
- `thinkdsp_gpu/__init__.py` - Package init (NEW)

### Configuration
- `environment.yml` - Add GPU dependencies
- `requirements.txt` - Add optional GPU deps
- `pyproject.toml` - Update dependencies

### Documentation
- `README.md` - GPU installation and usage
- `docs/GPU_MIGRATION.md` - This file
- `docs/PERFORMANCE.md` - Benchmark results

### Tests
- `tests/test_backend.py` - Backend tests (NEW)
- `tests/test_correctness.py` - CPU/GPU correctness (NEW)
- `tests/test_performance.py` - Performance benchmarks (NEW)

## Known Limitations

1. **DCT Implementation**: cuSignal may not have DCT, may need CuPy-based implementation
2. **scipy.stats**: Some statistical operations remain CPU-only (scipy dependency)
3. **File I/O**: WAV file reading/writing requires CPU arrays
4. **Small Arrays**: GPU overhead may not benefit very small signals (< 1K samples)

## Performance Expectations

- **Large FFTs** (> 10K samples): 5-50x speedup expected
- **Convolutions**: 3-20x speedup depending on kernel size
- **Spectrograms**: 5-30x speedup for large signals
- **Small operations**: May be slower due to GPU overhead

## Backward Compatibility

- All existing code continues to work
- CPU path is default if GPU unavailable
- No breaking changes to public API
- Notebooks work in both CPU and GPU modes

