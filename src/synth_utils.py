# --- Helpers audio généraux (swing, reverb, sidechain, etc.) ---

import numpy as np
import math

def db(x):
    """Convert dB to linear amplitude."""
    return 10 ** (x / 20)

def apply_env(x, env):
    n = min(len(x), len(env))
    y = np.copy(x[:n])
    y *= env[:n]
    if len(x) > n:
        y = np.concatenate([y, x[n:]])
    return y

def ducking_envelope_from_kick(total_len, fs, kick_positions, duck_ms=120, hold_ms=80, rel_ms=220, depth_db=-8):
    """Build a gain envelope that dips at every kick (classic sidechain)."""
    env = np.ones(int(total_len * fs))
    depth = db(depth_db)
    duck = int(duck_ms * fs / 1000)
    hold = int(hold_ms * fs / 1000)
    rel  = int(rel_ms  * fs / 1000)
    for pos in kick_positions:
        start = pos
        end_duck = min(start + duck, len(env))
        end_hold = min(end_duck + hold, len(env))
        end_rel  = min(end_hold + rel, len(env))
        if start < len(env):
            env[start:end_duck] *= np.linspace(1.0, depth, end_duck - start, endpoint=False)
        env[end_duck:end_hold] *= depth
        if end_rel > end_hold:
            env[end_hold:end_rel] *= np.linspace(depth, 1.0, end_rel - end_hold, endpoint=False)
    return env

def simple_delay_reverb(x, fs, delay_ms=70, feedback=0.25, mix=0.18):
    """Very simple mono FDN-like: single feedback delay (clean & CPU-cheap)."""
    d = int(delay_ms * fs / 1000)
    y = np.copy(x)
    buf = np.zeros(len(x) + d + 1)
    buf[:len(x)] = x
    for i in range(len(x)):
        delayed = buf[i]
        out = x[i] + feedback * delayed
        buf[i + d] += out * feedback
        y[i] = (1 - mix) * x[i] + mix * delayed
    return y

def lowpass_1pole(x, fs, cutoff_hz):
    """Tiny 1-pole low-pass for breakdown filter sweeps."""
    y = np.zeros_like(x)
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = dt / (rc + dt)
    for i in range(len(x)):
        y[i] = y[i-1] + alpha * (x[i] - y[i-1]) if i > 0 else alpha * x[i]
    return y

def jitter_indices(idxs, max_jitter_ms, fs, seed=7):
    """Humanize: shift indices a few ms randomly."""
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-int(max_jitter_ms*fs/1000), int(max_jitter_ms*fs/1000)+1, size=len(idxs))
    return np.clip(idxs + jitter, 0, None)
