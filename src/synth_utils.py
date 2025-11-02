import numpy as np
from src.signal_generators import sine_wave, square_wave, triangle_wave, sawtooth_wave
from src.envelope_adsr import adsr_envelope, apply_envelope
from src.filters import design_filter, apply_filter
from src.utils import FS, normalize_audio

def mix_signals(signals, weights=None, mode="auto"):
    """
    Mix or concatenate multiple audio signals.

    Parameters
    ----------
    signals : list of ndarrays
        List of audio signals (1D numpy arrays).
    weights : list or None
        Optional weights for mixing.
    mode : str
        'add'  → mix signals by addition (must have same length)
        'concat' → concatenate sequentially
        'auto' → automatically choose based on signal lengths

    Returns
    -------
    y : ndarray
        The mixed or concatenated signal.
    """
    lengths = [len(s) for s in signals]

    # Auto mode: decide based on length equality
    if mode == "auto":
        mode = "add" if len(set(lengths)) == 1 else "concat"

    if mode == "add":
        max_len = max(lengths)
        if weights is None:
            weights = np.ones(len(signals))
        acc = np.zeros(max_len)
        for s, w in zip(signals, weights):
            y = np.zeros(max_len)
            y[:len(s)] = s
            acc += w * y
        return normalize_audio(acc)

    elif mode == "concat":
        return normalize_audio(np.concatenate(signals))

    else:
        raise ValueError("mode must be 'add', 'concat', or 'auto'")


def multiply_signals(a, b):
    """Amplitude modulation (ring modulation)."""
    n = min(len(a), len(b))
    return a[:n] * b[:n]

def synthesize_sound(kind="techno", fs=FS):
    """
    Example synthesis chain combining oscillators, envelope, and filters.

    kind : str
        'techno', 'laser', 'bass', 'ambient', 'explosion', etc.

    Returns
    -------
    y : ndarray
        The synthesized waveform (float, normalized).
    """

    if kind == "techno":
        # Kick-like: sine base + envelope + lowpass
        x = sine_wave(60, duration=0.8, fs=fs)
        env, _ = adsr_envelope(attack=0.01, decay=0.1, sustain_level=0.0, sustain_time=0.0, release=0.4, fs=fs)
        y = apply_envelope(x, env)
        b, a = design_filter('lowpass', cutoff=800, fs=fs, order=4)
        y = apply_filter(y, b, a)
        return normalize_audio(y)

    elif kind == "laser":
        # Frequency-modulated chirp + short envelope
        t = np.arange(int(fs * 0.6)) / fs
        f0, f1 = 1000, 100
        x = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / t[-1]) * t)
        env, _ = adsr_envelope(attack=0.01, decay=0.05, sustain_level=0.0, sustain_time=0.0, release=0.1, fs=fs)
        y = apply_envelope(x, env)
        return normalize_audio(y)

    elif kind == "bass":
        # Sawtooth + lowpass + envelope
        x = sawtooth_wave(110, duration=1.0, fs=fs)
        b, a = design_filter('lowpass', cutoff=600, fs=fs, order=4)
        x = apply_filter(x, b, a)
        env, _ = adsr_envelope(attack=0.02, decay=0.1, sustain_level=0.7, sustain_time=0.4, release=0.3, fs=fs)
        y = apply_envelope(x, env)
        return normalize_audio(y)

    elif kind == "explosion":
        # Noise + all-pass filter cascade + envelope
        noise = np.random.randn(int(fs * 1.2))
        b, a = design_filter('allpass', cutoff=800, fs=fs, order=1)
        for _ in range(3):  # cascade for texture
            noise = apply_filter(noise, b, a)
        env, _ = adsr_envelope(attack=0.01, decay=0.3, sustain_level=0.0, sustain_time=0.0, release=0.7, fs=fs)
        y = apply_envelope(noise, env)
        return normalize_audio(y)

    elif kind == "ambient":
        # Add multiple detuned sine waves and slow envelope
        freqs = [220, 223, 226]
        waves = [sine_wave(f, duration=3.0, fs=fs) for f in freqs]
        mix = mix_signals(waves)
        env, _ = adsr_envelope(attack=1.0, decay=1.0, sustain_level=0.8, sustain_time=1.0, release=1.0, fs=fs)
        y = apply_envelope(mix, env)
        return normalize_audio(y)

    else:
        raise ValueError(f"Unknown sound type '{kind}'")
