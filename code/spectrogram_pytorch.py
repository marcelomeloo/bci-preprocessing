import numpy as np
import torch
import torch.nn.functional as F
import time

def spectrogram_pytorch(data, fs=250, nperseg=256, noverlap=None, window='hann'):
    """
    GPU-accelerated spectrogram using PyTorch.
    
    Parameters:
    -----------
    data : numpy.ndarray or torch.Tensor
        Input signal data
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment
    noverlap : int, optional
        Number of points to overlap between segments
    window : str
        Window function ('hann', 'hamming', 'blackman', etc.)
    
    Returns:
    --------
    f : numpy.ndarray
        Array of sample frequencies
    t : numpy.ndarray
        Array of segment times
    Sxx : numpy.ndarray
        Spectrogram of the input signal
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to PyTorch tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(device)
    else:
        data_tensor = data.float().to(device)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute STFT using PyTorch
    stft_result = torch.stft(
        data_tensor,
        n_fft=nperseg,
        hop_length=nperseg - noverlap,
        win_length=nperseg,
        window=torch.hann_window(nperseg).to(device),
        return_complex=True
    )
    
    # Calculate power spectral density
    Sxx_tensor = torch.abs(stft_result) ** 2
    
    # Create frequency and time arrays
    f = torch.fft.fftfreq(nperseg, 1/fs)[:nperseg//2 + 1]
    hop_length = nperseg - noverlap
    t = torch.arange(0, len(data_tensor) - nperseg + 1, hop_length) / fs
    
    # Convert back to numpy
    f_np = f.cpu().numpy()
    t_np = t.cpu().numpy()
    Sxx_np = Sxx_tensor.cpu().numpy()
    
    return f_np, t_np, Sxx_np


def stft_pytorch(data, fs=250, nperseg=256, noverlap=None):
    """
    GPU-accelerated STFT using PyTorch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(device)
    else:
        data_tensor = data.float().to(device)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    stft_result = torch.stft(
        data_tensor,
        n_fft=nperseg,
        hop_length=nperseg - noverlap,
        win_length=nperseg,
        window=torch.hann_window(nperseg).to(device),
        return_complex=True
    )
    
    f = torch.fft.fftfreq(nperseg, 1/fs)[:nperseg//2 + 1]
    hop_length = nperseg - noverlap
    t = torch.arange(0, len(data_tensor) - nperseg + 1, hop_length) / fs
    
    return f.cpu().numpy(), t.cpu().numpy(), stft_result.cpu().numpy()


def batch_spectrogram_pytorch(data_batch, fs=250, nperseg=256, noverlap=None):
    """
    Process multiple signals in batch using PyTorch.
    
    Parameters:
    -----------
    data_batch : numpy.ndarray or torch.Tensor
        Batch of signals with shape (batch_size, signal_length)
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
    Sxx_batch : numpy.ndarray
        Batch of spectrograms with shape (batch_size, freq_bins, time_bins)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(data_batch, np.ndarray):
        data_tensor = torch.from_numpy(data_batch).float().to(device)
    else:
        data_tensor = data_batch.float().to(device)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    batch_size, signal_length = data_tensor.shape
    
    # Process entire batch at once
    stft_results = []
    for i in range(batch_size):
        stft_result = torch.stft(
            data_tensor[i],
            n_fft=nperseg,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            window=torch.hann_window(nperseg).to(device),
            return_complex=True
        )
        stft_results.append(stft_result)
    
    # Stack results
    stft_batch = torch.stack(stft_results, dim=0)
    Sxx_batch = torch.abs(stft_batch) ** 2
    
    # Create frequency and time arrays
    f = torch.fft.fftfreq(nperseg, 1/fs)[:nperseg//2 + 1]
    hop_length = nperseg - noverlap
    t = torch.arange(0, signal_length - nperseg + 1, hop_length) / fs
    
    return f.cpu().numpy(), t.cpu().numpy(), Sxx_batch.cpu().numpy()


def compare_pytorch_methods(signal_length=10000, fs=250):
    """
    Compare PyTorch spectrogram with other methods.
    """
    # Generate test signal
    t = np.linspace(0, signal_length/fs, signal_length)
    test_signal = (np.sin(2*np.pi*10*t) + 
                  0.5*np.sin(2*np.pi*25*t) + 
                  0.2*np.random.randn(len(t)))
    
    print(f"Computing spectrogram with PyTorch (GPU: {torch.cuda.is_available()})...")
    
    start_time = time.time()
    f_torch, t_torch, Sxx_torch = spectrogram_pytorch(test_signal, fs=fs)
    pytorch_time = time.time() - start_time
    
    print(f"PyTorch time: {pytorch_time:.4f}s")
    print(f"Spectrogram shape: {Sxx_torch.shape}")
    print(f"Frequency range: {f_torch[0]:.2f} to {f_torch[-1]:.2f} Hz")
    
    return f_torch, t_torch, Sxx_torch


def eeg_spectrogram_pipeline_pytorch(eeg_data, fs=250, channels=None):
    """
    Complete EEG spectrogram processing pipeline using PyTorch.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (channels, samples) or (samples,) for single channel
    fs : float
        Sampling frequency
    channels : list, optional
        List of channel indices to process
    
    Returns:
    --------
    spectrograms : dict
        Dictionary containing spectrogram data for each channel
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if eeg_data.ndim == 1:
        # Single channel
        eeg_data = eeg_data.reshape(1, -1)
    
    if channels is None:
        channels = list(range(eeg_data.shape[0]))
    
    spectrograms = {}
    
    # Process in batch for efficiency
    batch_data = eeg_data[channels, :]
    f, t, Sxx_batch = batch_spectrogram_pytorch(batch_data, fs=fs)
    
    for i, ch_idx in enumerate(channels):
        spectrograms[f'channel_{ch_idx}'] = {
            'frequency': f,
            'time': t,
            'spectrogram': Sxx_batch[i]
        }
    
    return spectrograms


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! Running PyTorch GPU tests...")
        
        # Test single signal
        f, t, Sxx = compare_pytorch_methods(signal_length=20000)
        
        # Test batch processing
        print("\nTesting batch processing...")
        batch_signals = np.random.randn(5, 10000)  # 5 signals
        f_batch, t_batch, Sxx_batch = batch_spectrogram_pytorch(batch_signals)
        print(f"Batch spectrogram shape: {Sxx_batch.shape}")
        
    else:
        print("CUDA not available. PyTorch will use CPU.")
        f, t, Sxx = compare_pytorch_methods(signal_length=5000)
