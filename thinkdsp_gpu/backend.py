"""Backend abstraction layer for CPU/GPU operations.

This module provides a unified interface for CPU (NumPy/SciPy) and GPU (CuPy/cuSignal)
operations, with automatic detection and fallback to CPU when GPU is unavailable.
"""

import os
import warnings

import numpy as np

# Try to import GPU libraries
try:
    import cupy as cp
    import cusignal
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    cusignal = None
    GPU_AVAILABLE = False

# Backend selection via environment variable
_BACKEND_OVERRIDE = os.environ.get("THINKDSP_BACKEND", "").lower()


class Backend:
    """Backend abstraction for CPU/GPU operations.
    
    This class provides a unified interface for array operations, FFT,
    convolution, and window functions that work on both CPU and GPU.
    """
    
    def __init__(self, use_gpu=None):
        """Initialize backend.
        
        Args:
            use_gpu: If True, use GPU; if False, use CPU; if None, auto-detect.
        """
        if use_gpu is None:
            # Auto-detect: use GPU if available and not overridden
            if _BACKEND_OVERRIDE == "cpu":
                use_gpu = False
            elif _BACKEND_OVERRIDE == "gpu":
                use_gpu = True
            else:
                use_gpu = GPU_AVAILABLE
        
        if use_gpu and not GPU_AVAILABLE:
            warnings.warn(
                "GPU requested but not available. Falling back to CPU.",
                UserWarning
            )
            use_gpu = False
        
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        
        # Store original modules for reference
        self._np = np
        self._cp = cp
        
    @property
    def name(self):
        """Backend name."""
        return "gpu" if self.use_gpu else "cpu"
    
    def asarray(self, x, dtype=None):
        """Convert to backend array.
        
        Args:
            x: Array-like input
            dtype: Optional dtype
            
        Returns:
            Backend array (CuPy or NumPy)
        """
        if self.use_gpu:
            if isinstance(x, np.ndarray):
                return cp.asarray(x, dtype=dtype)
            elif isinstance(x, cp.ndarray):
                return x if dtype is None else x.astype(dtype)
            else:
                return cp.asarray(x, dtype=dtype)
        else:
            return np.asarray(x, dtype=dtype)
    
    def to_cpu(self, x):
        """Convert array to CPU (NumPy).
        
        Args:
            x: Array (CPU or GPU)
            
        Returns:
            NumPy array
        """
        if isinstance(x, np.ndarray):
            return x
        elif self._cp is not None and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        else:
            return np.asarray(x)
    
    def to_gpu(self, x):
        """Convert array to GPU (CuPy).
        
        Args:
            x: Array (CPU or GPU)
            
        Returns:
            CuPy array (if GPU available), otherwise NumPy array
        """
        if not self.use_gpu or self._cp is None:
            return np.asarray(x)
        
        if isinstance(x, cp.ndarray):
            return x
        elif isinstance(x, np.ndarray):
            return cp.asarray(x)
        else:
            return cp.asarray(x)
    
    def is_gpu_array(self, x):
        """Check if array is on GPU.
        
        Args:
            x: Array to check
            
        Returns:
            True if GPU array, False otherwise
        """
        return self._cp is not None and isinstance(x, cp.ndarray)
    
    # FFT operations
    def fft(self, x, n=None, axis=-1, norm=None):
        """FFT operation."""
        return self.xp.fft.fft(x, n=n, axis=axis, norm=norm)
    
    def ifft(self, x, n=None, axis=-1, norm=None):
        """IFFT operation."""
        return self.xp.fft.ifft(x, n=n, axis=axis, norm=norm)
    
    def rfft(self, x, n=None, axis=-1, norm=None):
        """Real FFT operation."""
        return self.xp.fft.rfft(x, n=n, axis=axis, norm=norm)
    
    def irfft(self, x, n=None, axis=-1, norm=None):
        """Inverse real FFT operation."""
        return self.xp.fft.irfft(x, n=n, axis=axis, norm=norm)
    
    def fftfreq(self, n, d=1.0):
        """FFT frequency array."""
        return self.xp.fft.fftfreq(n, d=d)
    
    def rfftfreq(self, n, d=1.0):
        """Real FFT frequency array."""
        return self.xp.fft.rfftfreq(n, d=d)
    
    def fftshift(self, x, axes=None):
        """FFT shift."""
        return self.xp.fft.fftshift(x, axes=axes)
    
    def ifftshift(self, x, axes=None):
        """Inverse FFT shift."""
        return self.xp.fft.ifftshift(x, axes=axes)
    
    # Convolution
    def convolve(self, a, v, mode='full'):
        """Convolution operation.
        
        Uses cuSignal if available on GPU, otherwise NumPy/CuPy.
        """
        if self.use_gpu and cusignal is not None:
            # Ensure arrays are on GPU
            a_gpu = self.to_gpu(a)
            v_gpu = self.to_gpu(v)
            return cusignal.convolve(a_gpu, v_gpu, mode=mode)
        else:
            return np.convolve(self.to_cpu(a), self.to_cpu(v), mode=mode)
    
    def fftconvolve(self, in1, in2, mode='full'):
        """FFT-based convolution.
        
        Uses cuSignal if available on GPU, otherwise scipy.signal.
        """
        if self.use_gpu and cusignal is not None:
            in1_gpu = self.to_gpu(in1)
            in2_gpu = self.to_gpu(in2)
            return cusignal.fftconvolve(in1_gpu, in2_gpu, mode=mode)
        else:
            # Fallback to scipy
            import scipy.signal
            return scipy.signal.fftconvolve(
                self.to_cpu(in1), self.to_cpu(in2), mode=mode
            )
    
    # Window functions
    def hamming(self, M):
        """Hamming window."""
        if self.use_gpu and cusignal is not None:
            return cusignal.windows.hamming(M)
        else:
            return np.hamming(M)
    
    def hanning(self, M):
        """Hanning window."""
        if self.use_gpu and cusignal is not None:
            return cusignal.windows.hanning(M)
        else:
            return np.hanning(M)
    
    def blackman(self, M):
        """Blackman window."""
        if self.use_gpu and cusignal is not None:
            return cusignal.windows.blackman(M)
        else:
            return np.blackman(M)
    
    def bartlett(self, M):
        """Bartlett window."""
        if self.use_gpu and cusignal is not None:
            return cusignal.windows.bartlett(M)
        else:
            return np.bartlett(M)
    
    def gaussian(self, M, std):
        """Gaussian window.
        
        Args:
            M: Window length
            std: Standard deviation
        """
        if self.use_gpu and cusignal is not None:
            return cusignal.windows.gaussian(M, std=std)
        else:
            import scipy.signal
            return scipy.signal.windows.gaussian(M, std=std)
    
    # DCT operations
    def dct(self, x, type=2, n=None, axis=-1, norm=None):
        """Discrete Cosine Transform.
        
        Note: cuSignal may not have DCT, so we implement using FFT.
        """
        if self.use_gpu and cusignal is not None:
            # Try cuSignal DCT first
            try:
                return cusignal.dct(x, type=type, n=n, axis=axis, norm=norm)
            except (AttributeError, NotImplementedError):
                # Fallback to FFT-based DCT implementation
                return _dct_fft_impl(self, x, type=type, n=n, axis=axis, norm=norm)
        else:
            import scipy.fftpack
            return scipy.fftpack.dct(x, type=type, n=n, axis=axis, norm=norm)
    
    def idct(self, x, type=2, n=None, axis=-1, norm=None):
        """Inverse Discrete Cosine Transform."""
        if self.use_gpu and cusignal is not None:
            try:
                return cusignal.idct(x, type=type, n=n, axis=axis, norm=norm)
            except (AttributeError, NotImplementedError):
                return _idct_fft_impl(self, x, type=type, n=n, axis=axis, norm=norm)
        else:
            import scipy.fftpack
            return scipy.fftpack.idct(x, type=type, n=n, axis=axis, norm=norm)


def _dct_fft_impl(backend, x, type=2, n=None, axis=-1, norm=None):
    """DCT implementation using FFT (for GPU when cuSignal DCT unavailable)."""
    x = backend.asarray(x)
    N = x.shape[axis] if n is None else n
    
    if type == 2:
        # Type-II DCT: DCT-II
        # Implementation using FFT
        if n is not None and n != x.shape[axis]:
            if n > x.shape[axis]:
                pad_shape = list(x.shape)
                pad_shape[axis] = n - x.shape[axis]
                x = backend.xp.concatenate([x, backend.xp.zeros(pad_shape)], axis=axis)
            else:
                x = backend.xp.take(x, range(n), axis=axis)
        
        # DCT-II via FFT
        N = x.shape[axis]
        # Create extended signal
        y = backend.xp.zeros((*x.shape[:axis], 2*N, *x.shape[axis+1:]))
        y[..., :N] = x
        y[..., N:] = x[..., ::-1]
        
        # FFT
        Y = backend.fft(y, axis=axis)
        
        # Extract and scale
        k = backend.xp.arange(N)
        factor = backend.xp.exp(-1j * backend.xp.pi * k / (2 * N))
        result = backend.xp.real(Y[..., :N] * factor)
        
        # Normalization
        if norm == 'ortho':
            result[..., 0] /= backend.xp.sqrt(2)
            result *= backend.xp.sqrt(2 / N)
        else:
            result[..., 0] /= 2
            result *= 2 / N
        
        return result
    else:
        # For other types, fall back to CPU
        import scipy.fftpack
        return scipy.fftpack.dct(backend.to_cpu(x), type=type, n=n, axis=axis, norm=norm)


def _idct_fft_impl(backend, x, type=2, n=None, axis=-1, norm=None):
    """IDCT implementation using FFT."""
    if type == 2:
        # Inverse of DCT-II is IDCT-II
        x = backend.asarray(x)
        N = x.shape[axis] if n is None else n
        
        # Normalization adjustment
        if norm == 'ortho':
            x = x.copy()
            x[..., 0] *= backend.xp.sqrt(2)
            x *= backend.xp.sqrt(N / 2)
        else:
            x = x.copy()
            x[..., 0] *= 2
            x *= N / 2
        
        # Create extended signal
        k = backend.xp.arange(N)
        factor = backend.xp.exp(1j * backend.xp.pi * k / (2 * N))
        X_ext = x * factor
        
        # Create full spectrum
        X_full = backend.xp.zeros((*x.shape[:axis], 2*N, *x.shape[axis+1:]), dtype=complex)
        X_full[..., :N] = X_ext
        X_full[..., N:] = backend.xp.conj(X_ext[..., ::-1])
        
        # IFFT
        y = backend.ifft(X_full, axis=axis)
        
        # Extract real part
        result = backend.xp.real(y[..., :N])
        
        return result
    else:
        import scipy.fftpack
        return scipy.fftpack.idct(backend.to_cpu(x), type=type, n=n, axis=axis, norm=norm)


# Global backend instance
_backend = None


def get_backend():
    """Get the current backend instance.
    
    Returns:
        Backend instance
    """
    global _backend
    if _backend is None:
        _backend = Backend()
    return _backend


def set_backend(use_gpu=None):
    """Set the backend.
    
    Args:
        use_gpu: If True, use GPU; if False, use CPU; if None, auto-detect.
    """
    global _backend
    _backend = Backend(use_gpu=use_gpu)
    return _backend


# Convenience functions
def to_cpu(x):
    """Convert array to CPU (NumPy)."""
    return get_backend().to_cpu(x)


def to_gpu(x):
    """Convert array to GPU (CuPy)."""
    return get_backend().to_gpu(x)


def is_gpu_array(x):
    """Check if array is on GPU."""
    return get_backend().is_gpu_array(x)

