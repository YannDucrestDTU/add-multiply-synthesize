# src/visualization.py  — core only
import numpy as np
import matplotlib.pyplot as plt

def db(x, floor_db=-140.0):
    """Convert magnitude array to dB with flooring."""
    with np.errstate(divide='ignore'):
        xd = 20.0 * np.log10(np.maximum(np.abs(x), 10**(floor_db/20)))
    return xd

def make_spectrum(x, fs):
    """Compute one-sided FFT magnitude spectrum."""
    N = len(x)
    Y = np.fft.rfft(x) / N
    f = np.fft.rfftfreq(N, d=1/fs)
    return Y, f

def plot_time(x, fs, tmax=0.01, title='Time domain'):
    """Plot first tmax seconds of the signal."""
    N = int(min(len(x), tmax * fs))
    t = np.arange(N) / fs
    plt.figure(figsize=(7, 2.4))
    plt.plot(t, x[:N])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_and_save_time(signal, fs, tmax, title, filename, folder="output/plots"):
    """Plot time-domain segment and save PNG; returns path."""
    from pathlib import Path
    Path(folder).mkdir(parents=True, exist_ok=True)
    N = int(tmax * fs)
    t = np.arange(N) / fs
    plt.figure(figsize=(7, 2.4))
    plt.plot(t, signal[:N])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    path = f"{folder}/{filename}"
    plt.savefig(path, dpi=300)
    plt.show()
    return path

def plot_spectrum(x, fs, title='Magnitude spectrum (dB)', xlim_hz=None):
    """Plot one-sided magnitude spectrum in dB."""
    Y, f = make_spectrum(x, fs)
    mag_db = db(np.abs(Y))
    plt.figure(figsize=(7, 2.8))
    plt.plot(f, mag_db)
    if xlim_hz:
        plt.xlim(0, xlim_hz)
    plt.ylim(-140, 10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB re 1)')
    plt.title(title)
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_and_save_spectrum(signal, fs, title, filename, fmax=5000, folder="output/plots"):
    """Plot single-sided magnitude spectrum and save PNG; returns path."""
    from pathlib import Path
    Path(folder).mkdir(parents=True, exist_ok=True)
    N = len(signal)
    X = np.fft.rfft(signal) / N
    f = np.fft.rfftfreq(N, d=1/fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    plt.figure(figsize=(7, 2.8))
    plt.plot(f, mag_db)
    plt.xlim(0, fmax)
    plt.ylim(-140, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    path = f"{folder}/{filename}"
    plt.savefig(path, dpi=300)
    plt.show()
    return path
