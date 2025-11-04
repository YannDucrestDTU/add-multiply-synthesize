import numpy as np
from src.utils import make_time, FS

def adsr_envelope(attack=0.1, decay=0.1, sustain_level=0.7, sustain_time=0.5, release=0.2, fs=48000):
    import numpy as np
    fs = int(fs)

    Na = int(round(fs*attack))
    Nd = int(round(fs*decay))
    Ns = int(round(fs*sustain_time))
    Nr = int(round(fs*release))

    a = np.linspace(0.0, 1.0, Na, endpoint=True) if Na > 0 else np.empty(0)
    d = (np.linspace(1.0, sustain_level, Nd+1, endpoint=True)[1:] if Nd > 0 else np.empty(0))
    s = (np.full(Ns, sustain_level) if Ns > 0 else np.empty(0))
    r = (np.linspace(sustain_level, 0.0, Nr+1, endpoint=True)[1:] if Nr > 0 else np.empty(0))

    env = np.concatenate((a, d, s, r))
    t = np.arange(env.size)/fs
    return env, t

def apply_envelope(signal, envelope):
    """
    Apply an envelope to a signal (elementwise multiplication).
    Truncate to shortest length if necessary.
    """
    n = min(len(signal), len(envelope))
    return signal[:n] * envelope[:n]
