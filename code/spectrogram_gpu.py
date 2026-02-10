import numpy as np
import cupy as cp
import cupyx.scipy.signal
from scipy import signal
import time
import matplotlib.pyplot as plt

def spectrogram_gpu_cupy(data, fs=250, nperseg=256, noverlap=None):
    """
    GPU-accelerated spectrogram using CuPy.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input signal data
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment
    noverlap : int, optional
        Number of points to overlap between segments
    
    Returns:
    --------
    f : numpy.ndarray
        Array of sample frequencies
    t : numpy.ndarray
        Array of segment times
    Sxx : numpy.ndarray
        Spectrogram of the input signal
    """
    # Convert to GPU
    data_gpu = cp.asarray(data)
    
    # Compute spectrogram on GPU
    f_gpu, t_gpu, Sxx_gpu = cupyx.scipy.signal.spectrogram(
        data_gpu, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    # Convert back to CPU
    f = cp.asnumpy(f_gpu)
    t = cp.asnumpy(t_gpu)
    Sxx = cp.asnumpy(Sxx_gpu)
    
    return f, t, Sxx


def spectrogram_cpu_scipy(data, fs=250, nperseg=256, noverlap=None):
    """
    CPU-based spectrogram using SciPy for comparison.
    """
    return signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)


def benchmark_spectrogram(signal_length=100000, fs=250):
    """
    Benchmark GPU vs CPU spectrogram computation.
    """
    # Generate test signal
    t = np.linspace(0, signal_length/fs, signal_length)
    test_signal = (np.sin(2*np.pi*10*t) + 
                  0.5*np.sin(2*np.pi*25*t) + 
                  0.2*np.random.randn(len(t)))
    
    print(f"Benchmarking spectrogram computation on signal of length {signal_length}...")
    
    # GPU benchmark
    start_time = time.time()
    f_gpu, t_gpu, Sxx_gpu = spectrogram_gpu_cupy(test_signal, fs=fs)
    gpu_time = time.time() - start_time
    
    # CPU benchmark
    start_time = time.time()
    f_cpu, t_cpu, Sxx_cpu = spectrogram_cpu_scipy(test_signal, fs=fs)
    cpu_time = time.time() - start_time
    
    # Verify results are similar
    freq_error = np.max(np.abs(f_gpu - f_cpu))
    time_error = np.max(np.abs(t_gpu - t_cpu))
    spec_error = np.max(np.abs(Sxx_gpu - Sxx_cpu))
    
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"Max frequency error: {freq_error:.2e}")
    print(f"Max time error: {time_error:.2e}")
    print(f"Max spectrogram error: {spec_error:.2e}")
    
    return f_gpu, t_gpu, Sxx_gpu


def stft_gpu_cupy(data, fs=250, nperseg=256, noverlap=None):
    """
    GPU-accelerated Short-Time Fourier Transform using CuPy.
    """
    data_gpu = cp.asarray(data)
    
    f_gpu, t_gpu, Zxx_gpu = cupyx.scipy.signal.stft(
        data_gpu,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    return cp.asnumpy(f_gpu), cp.asnumpy(t_gpu), cp.asnumpy(Zxx_gpu)


def process_eeg_channels_gpu(eeg_data, fs=250, channels=None):
    """
    Process multiple EEG channels with GPU acceleration.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency
    channels : list, optional
        List of channel indices to process
    
    Returns:
    --------
    results : dict
        Dictionary containing frequency, time, and spectrogram arrays for each channel
    """
    if channels is None:
        channels = list(range(eeg_data.shape[0]))
    
    results = {}
    
    print(f"Processing {len(channels)} EEG channels on GPU...")
    
    for i, ch_idx in enumerate(channels):
        channel_data = eeg_data[ch_idx, :]
        f, t, Sxx = spectrogram_gpu_cupy(channel_data, fs=fs)
        
        results[f'channel_{ch_idx}'] = {
            'frequency': f,
            'time': t,
            'spectrogram': Sxx
        }
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(channels)} channels")
    
    return results


def check_gpu_availability():
    """Check if GPU is available for CuPy operations."""
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False


if __name__ == "__main__":
    if not check_gpu_availability():
        print("Warning: GPU not available. CuPy operations may fall back to CPU.")
    else:
        print("GPU detected. Running benchmarks...")
        
        # Run benchmark
        f, t, Sxx = benchmark_spectrogram(signal_length=50000)
        
        print("\nSpectrogram shape:", Sxx.shape)
        print("Frequency range:", f[0], "to", f[-1], "Hz")
        print("Time range:", t[0], "to", t[-1], "s")
