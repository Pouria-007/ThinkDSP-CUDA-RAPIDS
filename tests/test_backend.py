"""Tests for the GPU backend abstraction layer."""

import numpy as np
import pytest

try:
    from thinkdsp_gpu.backend import Backend, get_backend, set_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    pytestmark = pytest.mark.skip("GPU backend not available")


def test_backend_initialization():
    """Test backend can be initialized."""
    backend = Backend(use_gpu=False)
    assert backend.name == "cpu"
    assert backend.xp is np


def test_backend_asarray():
    """Test array conversion."""
    backend = Backend(use_gpu=False)
    x = [1, 2, 3]
    arr = backend.asarray(x)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.array([1, 2, 3]))


def test_backend_fft():
    """Test FFT operations."""
    backend = Backend(use_gpu=False)
    x = np.array([1, 2, 3, 4])
    result = backend.fft(x)
    expected = np.fft.fft(x)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_backend_rfft():
    """Test real FFT operations."""
    backend = Backend(use_gpu=False)
    x = np.array([1, 2, 3, 4])
    result = backend.rfft(x)
    expected = np.fft.rfft(x)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_backend_convolve():
    """Test convolution."""
    backend = Backend(use_gpu=False)
    a = np.array([1, 2, 3])
    v = np.array([0.5, 0.5])
    result = backend.convolve(a, v, mode='full')
    expected = np.convolve(a, v, mode='full')
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_backend_hamming():
    """Test Hamming window."""
    backend = Backend(use_gpu=False)
    result = backend.hamming(10)
    expected = np.hamming(10)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_backend_to_cpu():
    """Test CPU conversion."""
    backend = Backend(use_gpu=False)
    x = np.array([1, 2, 3])
    result = backend.to_cpu(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, x)


def test_get_backend():
    """Test global backend getter."""
    backend = get_backend()
    assert isinstance(backend, Backend)


def test_set_backend():
    """Test global backend setter."""
    original = get_backend()
    set_backend(use_gpu=False)
    backend = get_backend()
    assert backend.name == "cpu"
    # Restore
    set_backend(use_gpu=original.use_gpu)

