import numpy as np
from src.signal_generators import sine_wave, square_wave, sawtooth_wave, white_noise
from src.utils import FS, normalize_audio, save_wav
from src.envelope_adsr import adsr_envelope as generate_adsr, apply_envelope
from src.filters import design_filter, apply_filter

fs = FS

# =========================
# 1) KICK — techno settings
# =========================
dur_kick = 0.16          # 120–180 ms → 160 ms
freq_kick = 60
kick = sine_wave(freq_kick, dur_kick, fs)

# ADSR: A=0–2 ms, D=100–150 ms, S=0, R=30–60 ms
env_kick_adsr, _ = generate_adsr(
    attack=0.002, decay=0.12,
    sustain_level=0.0, sustain_time=0.0,
    release=0.05, fs=fs
)
kick_adsr = apply_envelope(kick, env_kick_adsr)

# FILTRE: LPF 150–250 Hz (résonance ≈ 0.2). On garde un HPF léger pour le sub-rumble.
b_kick_hpf, a_kick_hpf = design_filter('highpass', cutoff=35, order=2, fs=fs)
b_kick_lpf, a_kick_lpf = design_filter('lowpass',  cutoff=200, order=2, fs=fs)  # ~200 Hz
kick_filt = apply_filter(apply_filter(kick_adsr, b_kick_hpf, a_kick_hpf), b_kick_lpf, a_kick_lpf)

# ============================
# 2) HI-HAT — closed & open
# ============================
# Durées: closed=60–120 ms → 90 ms ; open=400–600 ms → 500 ms
dur_hat_closed = 0.09
dur_hat_open   = 0.50
hihat_closed = white_noise(dur_hat_closed, fs)
hihat_open   = white_noise(dur_hat_open,   fs)

# ADSR closed: A=0–5 ms → 3 ms ; D=70–100 ms → 80 ms ; S=0 ; R=20–40 ms → 30 ms
env_hat_closed, _ = generate_adsr(
    attack=0.003, decay=0.08,
    sustain_level=0.0, sustain_time=0.0,
    release=0.03, fs=fs
)
# ADSR open: A=3 ms ; D=500 ms ; S=0 ; R=90 ms
env_hat_open, _ = generate_adsr(
    attack=0.003, decay=0.50,
    sustain_level=0.0, sustain_time=0.0,
    release=0.09, fs=fs
)

hihat_closed_adsr = apply_envelope(hihat_closed, env_hat_closed)
hihat_open_adsr   = apply_envelope(hihat_open,   env_hat_open)

# FILTRE: HPF 6–10 kHz → 8 kHz
b_hat_hpf, a_hat_hpf = design_filter('highpass', cutoff=8000, order=4, fs=fs)
hihat_closed_filt = apply_filter(hihat_closed_adsr, b_hat_hpf, a_hat_hpf)
hihat_open_filt   = apply_filter(hihat_open_adsr,   b_hat_hpf, a_hat_hpf)

# =========================
# 3) BASS — techno settings
# =========================
dur_bass = 0.60   # 300–800 ms → 600 ms
freq_bass = 110
bass = sawtooth_wave(freq_bass, dur_bass, fs)

# LFO d’amplitude doux (optionnel)
t_bass = np.arange(len(bass)) / fs
bass_lfo = 0.25 * (1 + np.sin(2 * np.pi * 1.0 * t_bass)) + 0.75  # 0.75–1.25
bass *= bass_lfo

# ADSR: A=10–30 ms → 20 ms ; D=200–400 ms → 300 ms ; S=0.6–0.8 → 0.7 ; R=150–300 ms → 200 ms
env_bass, _ = generate_adsr(
    attack=0.020, decay=0.30,
    sustain_level=0.70, sustain_time=max(0.0, dur_bass - (0.020 + 0.30 + 0.20)),
    release=0.20, fs=fs
)
bass_adsr = apply_envelope(bass, env_bass)

# FILTRE: LPF 120–400 Hz → 250 Hz (+ léger HPF anti-DC)
b_bass_hpf, a_bass_hpf = design_filter('highpass', cutoff=30, order=2, fs=fs)
b_bass_lpf, a_bass_lpf = design_filter('lowpass',  cutoff=250, order=2, fs=fs)
bass_filt = apply_filter(apply_filter(bass_adsr, b_bass_hpf, a_bass_hpf), b_bass_lpf, a_bass_lpf)

# =========================
# 4) PAD — techno settings (version adoucie)
# =========================
dur_pad = 6.0
freq_pad = 220.0

# Trois sinusoïdes avec très léger désaccordage pour éviter les battements marqués
pad1 = sine_wave(freq_pad,               dur_pad, fs)
pad2 = sine_wave(freq_pad * 1.003,       dur_pad, fs)  # +0.3 %
pad3 = sine_wave(freq_pad * (1.0 - 0.003), dur_pad, fs)  # -0.3 %
pad = (pad1 + pad2 + pad3) / 3.0

# LFO d'amplitude très discret et lent (±5% autour de 1.0)
t_pad = np.arange(len(pad)) / fs
pad_lfo = 0.05 * np.sin(2 * np.pi * 0.10 * t_pad) + 1.0  # 0.95–1.05 à 0.10 Hz
pad *= pad_lfo

# ADSR lent et fluide
env_pad, _ = generate_adsr(
    attack=1.50, decay=1.20,
    sustain_level=0.85,
    sustain_time=max(0.0, dur_pad - (1.50 + 1.20 + 3.00)),
    release=3.00, fs=fs
)
pad_adsr = apply_envelope(pad, env_pad)

# Filtrage : HPF léger pour le rumble + LPF plus bas pour lisser le timbre
b_pad_hpf, a_pad_hpf = design_filter('highpass', cutoff=30,   order=2, fs=fs)
b_pad_lpf, a_pad_lpf = design_filter('lowpass',  cutoff=800,  order=4, fs=fs)
pad_filt = apply_filter(pad_adsr, b_pad_hpf, a_pad_hpf)
pad_filt = apply_filter(pad_filt,  b_pad_lpf, a_pad_lpf)

# Micro-mouvement (très léger) + niveau faible pour rester en fond
pad_filt *= (0.98 + 0.02 * np.sin(2 * np.pi * 0.25 * np.arange(len(pad_filt)) / fs))
pad_filt *= 0.20



# =========================
# Export des stems + mix
# =========================
save_wav("output/sounds/step5_kick_filtered.wav",          normalize_audio(kick_filt),          fs)
save_wav("output/sounds/step5_hihat_closed_filtered.wav",  normalize_audio(hihat_closed_filt),  fs)
save_wav("output/sounds/step5_hihat_open_filtered.wav",    normalize_audio(hihat_open_filt),    fs)
save_wav("output/sounds/step5_bass_filtered.wav",          normalize_audio(bass_filt),          fs)
save_wav("output/sounds/step5_pad_filtered.wav",           normalize_audio(pad_filt),           fs)

# Mix: on additionne kick + closed hat + bass + pad (open hat laissé en stem séparé)
min_len = min(len(kick_filt), len(hihat_closed_filt), len(bass_filt), len(pad_filt))
mix = kick_filt[:min_len] + hihat_closed_filt[:min_len] + bass_filt[:min_len] + pad_filt[:min_len]
mix = normalize_audio(mix)
save_wav("output/sounds/step5_mix.wav", mix, fs)
import numpy as np
from src.utils import FS, normalize_audio, save_wav
import soundfile as sf

fs = FS
bpm = 125
beat_dur = 60 / bpm             # durée d’un temps
bars = 5                        # 5 mesures de 4 temps → 10 s environ
total_dur = 4 * bars * beat_dur

# Chargement des sons (assure-toi qu’ils sont dans output/sounds/)
kick, _ = sf.read("output/sounds/step5_kick_filtered.wav")
hat_c, _ = sf.read("output/sounds/step5_hihat_closed_filtered.wav")
hat_o, _ = sf.read("output/sounds/step5_hihat_open_filtered.wav")
bass, _ = sf.read("output/sounds/step5_bass_filtered.wav")
pad,  _ = sf.read("output/sounds/step5_pad_filtered.wav")

# Création de la timeline
t = np.arange(int(total_dur * fs)) / fs
mix = np.zeros_like(t)

# =========================
# Placement des éléments
# =========================

# --- Kick (4/4) ---
for i in range(4 * bars):
    start = int(i * beat_dur * fs)
    mix[start:start+len(kick)] += kick

# --- Hi-hat fermé (contretemps) ---
for i in range(4 * bars):
    start = int((i + 0.5) * beat_dur * fs)
    mix[start:start+len(hat_c)] += hat_c * 0.5  # plus doux

# --- Hi-hat ouvert (toutes les 2 mesures, sur le dernier temps) ---
for i in range(0, bars, 2):
    start = int(((i+1)*4 - 0.5) * beat_dur * fs)
    end = start + len(hat_o)
    if end > len(mix):
        end = len(mix)
    mix[start:end] += hat_o[:end - start] * 0.6


# --- Bass (1 note par temps, légère variation) ---
bass_note = bass * 0.8
for i in range(4 * bars):
    start = int(i * beat_dur * fs)
    end = start + len(bass_note)
    if end > len(mix):
        end = len(mix)
    mix[start:end] += bass_note[:end - start]


# --- Pad (en continu) ---
# Si le pad est plus court → on le répète jusqu’à couvrir toute la durée du mix
if len(pad) < len(mix):
    pad_stretch = np.tile(pad, int(np.ceil(len(mix) / len(pad))))[:len(mix)] * 0.4
else:
    pad_stretch = pad[:len(mix)] * 0.4

mix += pad_stretch


# =========================
# Normalisation et export
# =========================
mix = normalize_audio(mix)
save_wav("output/sounds/test_techno_10s.wav", mix, fs)
print("✅ Petite boucle techno de 10 secondes générée → output/sounds/test_techno_10s.wav")

