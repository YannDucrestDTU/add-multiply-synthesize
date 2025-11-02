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

def plot_and_save_time(signal, fs, tmax, title, filename, folder="output/plots"):
    """
    Plot a time-domain signal and save it as a PNG file.

    Parameters
    ----------
    signal : ndarray
        Input signal to plot.
    fs : int
        Sampling frequency (Hz).
    tmax : float
        Duration (s) of signal segment to display.
    title : str
        Title for the plot.
    filename : str
        Filename (e.g., "square_wave.png").
    folder : str
        Folder path to save the figure.

    Returns
    -------
    path : str
        Full path to the saved figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
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
    """
    Compute and plot the single-sided magnitude spectrum of a signal, then save it.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    fs : int
        Sampling frequency (Hz).
    title : str
        Title for the plot.
    filename : str
        Name of the PNG file to save.
    fmax : float, optional
        Maximum frequency to display (Hz).
    folder : str, optional
        Output folder for plots.

    Returns
    -------
    path : str
        Full path to the saved plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
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

def plot_and_save_response(b, a, fs, title, filename_prefix, folder="output/plots"):
    """
    Plot and save magnitude and phase responses of a digital filter.

    Parameters
    ----------
    b, a : ndarray
        Filter coefficients.
    fs : int
        Sampling rate (Hz).
    title : str
        Plot title prefix.
    filename_prefix : str
        Base name for saving (without extension).
    folder : str
        Output directory.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.signal import freqz

    Path(folder).mkdir(parents=True, exist_ok=True)

    # Frequency response
    w, h = freqz(b, a, worN=2048, fs=fs)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.unwrap(np.angle(h))

    # Magnitude plot
    plt.figure(figsize=(7, 2.8))
    plt.plot(w, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{title} – Magnitude response")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    mag_path = f"{folder}/{filename_prefix}_mag.png"
    plt.savefig(mag_path, dpi=300)
    plt.show()

    # Phase plot
    plt.figure(figsize=(7, 2.8))
    plt.plot(w, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.title(f"{title} – Phase response")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    phase_path = f"{folder}/{filename_prefix}_phase.png"
    plt.savefig(phase_path, dpi=300)
    plt.show()

    return mag_path, phase_path

def plot_and_save_spectrogram(signal, fs, filename, nfft=1024, noverlap=512, title="Spectrogram", folder="output/plots"):
    """
    Plot magnitude spectrogram of `signal` and save the figure.

    Parameters
    ----------
    signal : ndarray
        Input mono signal.
    fs : int
        Sampling frequency (Hz).
    filename : str
        Output PNG filename (e.g., "mix_spectrogram.png").
    nfft : int
        FFT size for the spectrogram.
    noverlap : int
        Number of points to overlap between segments.
    title : str
        Plot title.
    folder : str
        Folder to save the figure.

    Returns
    -------
    path : str
        Full path to the saved figure.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 3.2))
    plt.specgram(signal, NFFT=nfft, Fs=fs, noverlap=noverlap)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    path = f"{folder}/{filename}"
    plt.savefig(path, dpi=300)
    plt.show()
    return path
