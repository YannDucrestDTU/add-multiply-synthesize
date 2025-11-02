import numpy as np
from scipy import signal
from src.utils import make_time, FS, DEFAULT_DUR

def sine_wave(freq, duration=DEFAULT_DUR, amp=1.0, phase=0.0, fs=FS):
    """Pure sine wave generator."""
    t = make_time(duration, fs)
    return amp * np.sin(2 * np.pi * freq * t + phase)

def square_wave(freq, duration=DEFAULT_DUR, amp=1.0, duty=0.5, fs=FS):
    """Square wave using scipy.signal.square with adjustable duty cycle."""
    t = make_time(duration, fs)
    return amp * signal.square(2 * np.pi * freq * t, duty=duty)

def triangle_wave(freq, duration=DEFAULT_DUR, amp=1.0, fs=FS):
    """Triangle wave using sawtooth with width=0.5."""
    t = make_time(duration, fs)
    return amp * signal.sawtooth(2 * np.pi * freq * t, width=0.5)

def sawtooth_wave(freq, duration=DEFAULT_DUR, amp=1.0, width=1.0, fs=FS):
    """Sawtooth wave with selectable width (0.0–1.0)."""
    t = make_time(duration, fs)
    return amp * signal.sawtooth(2 * np.pi * freq * t, width=width)
