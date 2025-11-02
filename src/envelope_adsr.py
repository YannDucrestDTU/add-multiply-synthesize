import numpy as np
from src.utils import make_time, FS

def adsr_envelope(attack=0.1, decay=0.1, sustain_level=0.7, sustain_time=0.5, release=0.2, fs=FS):
    """
    Generate an ADSR envelope.

    Parameters
    ----------
    attack : float
        Attack duration in seconds (rise 0 → 1).
    decay : float
        Decay duration in seconds (1 → sustain_level).
    sustain_level : float
        Amplitude during sustain phase (0–1).
    sustain_time : float
        Sustain duration in seconds.
    release : float
        Release duration in seconds (fall to 0).
    fs : int
        Sampling frequency.

    Returns
    -------
    env : ndarray
        Envelope samples in range [0, 1].
    t : ndarray
        Time vector corresponding to env.
    """
    a = np.linspace(0, 1, int(fs * attack), endpoint=False)
    d = np.linspace(1, sustain_level, int(fs * decay), endpoint=False)
    s = np.ones(int(fs * sustain_time)) * sustain_level
    r = np.linspace(sustain_level, 0, int(fs * release))
    env = np.concatenate((a, d, s, r))
    t = np.arange(len(env)) / fs
    return env, t

def apply_envelope(signal, envelope):
    """
    Apply an envelope to a signal (elementwise multiplication).
    Truncate to shortest length if necessary.
    """
    n = min(len(signal), len(envelope))
    return signal[:n] * envelope[:n]
