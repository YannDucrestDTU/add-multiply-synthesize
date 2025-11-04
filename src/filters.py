import numpy as np
from scipy import signal
from src.utils import FS

def design_filter(kind='lowpass', cutoff=1000, fs=FS, order=4, band=None):
    if kind == 'bandpass':
        if band is None:
            raise ValueError("For bandpass, provide band=(low, high).")
        b, a = signal.butter(order, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    elif kind == 'allpass':
        # All-pass filter: same order, but zero-pole mirrored
        # Here we use a simple phase all-pass example (one-pole)
        w0 = 2 * np.pi * cutoff / fs
        alpha = (np.sin(w0) - np.cos(w0)) / (np.sin(w0) + np.cos(w0))
        b = np.array([alpha, 1])
        a = np.array([1, alpha])
    else:
        b, a = signal.butter(order, cutoff / (fs / 2), btype=kind)
    return b, a

def apply_filter(x, b, a):
    """Apply IIR/FIR filter to signal using zero-phase filtering."""
    return signal.filtfilt(b, a, x)

def freq_response(b, a, fs):
    """Compute magnitude and phase response."""
    w, h = signal.freqz(b, a, worN=2048, fs=fs)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.unwrap(np.angle(h))
    return w, mag, phase

