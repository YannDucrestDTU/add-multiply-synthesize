import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, group_delay

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

def plot_phase_response(b, a, fs, title="Phase response"):
    """Plot phase response (unwrapped) of a digital filter."""
    w, h = freqz(b, a, worN=2048, fs=fs)
    phase = np.unwrap(np.angle(h))
    plt.figure(figsize=(7, 2.6))
    plt.plot(w, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_group_delay(b, a, fs, title="Group delay"):
    """Plot group delay of a digital filter in samples."""
    w, gd = group_delay((b, a), fs=fs)
    plt.figure(figsize=(7, 2.6))
    plt.plot(w, gd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Group delay (samples)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(x, fs, nfft=1024, noverlap=512, title="Spectrogram"):
    """Plot magnitude spectrogram."""
    plt.figure(figsize=(7, 3.0))
    plt.specgram(x, NFFT=nfft, Fs=fs, noverlap=noverlap)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
