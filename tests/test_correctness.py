"""Correctness tests comparing CPU vs GPU outputs."""

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


@pytest.fixture
def test_signal():
    """Generate a test signal."""
    duration = 0.1
    framerate = 11025
    freq = 440
    n = int(duration * framerate)
    ts = np.arange(n) / framerate
    ys = np.sin(2 * np.pi * freq * ts)
    return ys, framerate


def test_fft_correctness(test_signal):
    """Test FFT correctness between CPU and GPU."""
    ys, framerate = test_signal
    
    # CPU result
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    spectrum_cpu = wave_cpu.make_spectrum()
    hs_cpu = spectrum_cpu.hs
    fs_cpu = spectrum_cpu.fs
    
    # GPU result (if available)
    backend = get_backend()
    if backend.use_gpu:
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        spectrum_gpu = wave_gpu.make_spectrum()
        hs_gpu = backend.to_cpu(spectrum_gpu.hs)
        fs_gpu = backend.to_cpu(spectrum_gpu.fs)
        
        # Compare results
        np.testing.assert_allclose(hs_cpu, hs_gpu, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(fs_cpu, fs_gpu, rtol=1e-10)
    
    # Restore CPU backend
    set_backend(use_gpu=False)


def test_ifft_correctness(test_signal):
    """Test IFFT correctness."""
    ys, framerate = test_signal
    
    # CPU: Wave -> Spectrum -> Wave
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    spectrum_cpu = wave_cpu.make_spectrum()
    wave_cpu_recon = spectrum_cpu.make_wave()
    
    backend = get_backend()
    if backend.use_gpu:
        # GPU: Wave -> Spectrum -> Wave
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        spectrum_gpu = wave_gpu.make_spectrum()
        wave_gpu_recon = spectrum_gpu.make_wave()
        
        # Compare reconstructed waves
        np.testing.assert_allclose(
            wave_cpu_recon.ys, wave_gpu_recon.ys, rtol=1e-5, atol=1e-7
        )
    
    set_backend(use_gpu=False)


def test_convolution_correctness(test_signal):
    """Test convolution correctness."""
    ys, framerate = test_signal
    
    # CPU convolution
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    window = np.array([0.25, 0.5, 0.25])
    convolved_cpu = wave_cpu.convolve(window)
    
    backend = get_backend()
    if backend.use_gpu:
        # GPU convolution
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        convolved_gpu = wave_gpu.convolve(window)
        
        # Compare results
        np.testing.assert_allclose(
            convolved_cpu.ys, convolved_gpu.ys, rtol=1e-5, atol=1e-7
        )
    
    set_backend(use_gpu=False)


def test_window_correctness(test_signal):
    """Test window function correctness."""
    ys, framerate = test_signal
    
    # CPU windowing
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    wave_cpu.hamming()
    result_cpu = wave_cpu.ys
    
    backend = get_backend()
    if backend.use_gpu:
        # GPU windowing
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        wave_gpu.hamming()
        result_gpu = wave_gpu.ys
        
        # Compare results
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5, atol=1e-7)
    
    set_backend(use_gpu=False)


def test_dct_correctness(test_signal):
    """Test DCT correctness."""
    ys, framerate = test_signal
    
    # CPU DCT
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    dct_cpu = wave_cpu.make_dct()
    hs_cpu = dct_cpu.hs
    
    backend = get_backend()
    if backend.use_gpu:
        # GPU DCT
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        dct_gpu = wave_gpu.make_dct()
        hs_gpu = backend.to_cpu(dct_gpu.hs)
        
        # Compare results (DCT may have larger numerical differences)
        np.testing.assert_allclose(hs_cpu, hs_gpu, rtol=1e-4, atol=1e-6)
    
    set_backend(use_gpu=False)


def test_spectrogram_correctness(test_signal):
    """Test spectrogram correctness."""
    ys, framerate = test_signal
    
    # CPU spectrogram
    set_backend(use_gpu=False)
    wave_cpu = thinkdsp.Wave(ys, framerate=framerate)
    specgram_cpu = wave_cpu.make_spectrogram(seg_length=512)
    
    backend = get_backend()
    if backend.use_gpu:
        # GPU spectrogram
        set_backend(use_gpu=True)
        wave_gpu = thinkdsp.Wave(ys, framerate=framerate)
        specgram_gpu = wave_gpu.make_spectrogram(seg_length=512)
        
        # Compare spectrograms
        times_cpu = specgram_cpu.times()
        times_gpu = specgram_gpu.times()
        assert len(times_cpu) == len(times_gpu)
        
        # Compare a few spectra
        for t in times_cpu[:3]:  # Compare first 3
            spec_cpu = specgram_cpu.spec_map[t]
            spec_gpu = specgram_gpu.spec_map[t]
            amps_cpu = spec_cpu.amps
            amps_gpu = backend.to_cpu(spec_gpu.amps)
            np.testing.assert_allclose(amps_cpu, amps_gpu, rtol=1e-4, atol=1e-6)
    
    set_backend(use_gpu=False)

