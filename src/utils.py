import numpy as np
from scipy.io.wavfile import write

FS = 44_100  # Default sampling rate (Hz)
DEFAULT_DUR = 1.0  # Default duration in seconds

def make_time(duration=DEFAULT_DUR, fs=FS):
    """Return time vector t with step 1/fs and length 'duration' (no endpoint)."""
    n = int(np.floor(duration * fs))
    return np.arange(n) / fs

def normalize_audio(x, peak=0.99):
    """Peak-normalize to avoid clipping when saving. Keeps shape, returns float64."""
    m = np.max(np.abs(x)) + 1e-12
    return (x / m) * peak

def save_wav(path, x, fs=FS):
    """Save mono float signal to 16-bit PCM WAV."""
    x16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
    write(path, fs, x16)

def save_norm(path, x, fs=FS, peak=0.99):
    """Normalize then save mono float signal to 16-bit PCM WAV."""
    x_n = normalize_audio(x, peak=peak)
    save_wav(path, x_n, fs)
