import numpy as np
from src.signal_generators import sine_wave, square_wave, sawtooth_wave, white_noise
from src.envelope_adsr import adsr_envelope as gen_adsr, apply_envelope
from src.filters import design_filter, apply_filter
from src.utils import FS, normalize_audio, save_wav

fs = FS
dur = 2.0  # durée de test (2 s)

# =====================================================
# 🔧 Choisis ici les paramètres pour CHAQUE SON
# =====================================================

# --- KICK ---
kick_wave   = sine_wave(60, dur, fs)                     # <--- change la wave ici
kick_adsr   = dict(attack=0.002, decay=0.12, sustain_level=0.0, release=0.05)
kick_filter = dict(type="lowpass", cutoff=200, order=2)  # <--- change ou mets None

# --- HI-HAT ---
hihat_wave   = white_noise(dur, fs)
hihat_adsr   = dict(attack=0.003, decay=0.08, sustain_level=0.0, release=0.03)
hihat_filter = dict(type="highpass", cutoff=8000, order=4)

# --- BASS ---
bass_wave   = sawtooth_wave(110, dur, fs)
bass_adsr   = dict(attack=0.02, decay=0.3, sustain_level=0.7, release=0.2)
bass_filter = dict(type="lowpass", cutoff=200, order=2)

# --- PAD ---
pad_wave   = sine_wave(220, dur, fs)
pad_adsr   = dict(attack=0.8, decay=0.8, sustain_level=0.85, release=1.5)
pad_filter = dict(type="lowpass", cutoff=800, order=4)

# =====================================================
# ⚙️ Fonction utilitaire pour tout tester
# =====================================================
def make_sound(wave, name, adsr, filt=None):
    env, _ = gen_adsr(
        attack=adsr.get("attack",0.01),
        decay=adsr.get("decay",0.1),
        sustain_level=adsr.get("sustain_level",0.7),
        sustain_time=1.0,
        release=adsr.get("release",0.2),
        fs=fs
    )
    y = apply_envelope(wave, env)
    if filt is not None:
        if filt["type"] == "bandpass":
            b, a = design_filter(filt["type"], band=filt["band"], order=filt.get("order",2), fs=fs)
        else:
            b, a = design_filter(filt["type"], cutoff=filt["cutoff"], order=filt.get("order",2), fs=fs)
        y = apply_filter(y, b, a)
    y = normalize_audio(y)
    save_wav(f"output/{name}_test.wav", y, fs)
    print(f"✅ {name} → output/{name}_test.wav")

# =====================================================
# 🎧 Génère chaque son (2 s chacun)
# =====================================================
make_sound(kick_wave,  "kick",  kick_adsr,  kick_filter)
make_sound(hihat_wave, "hihat", hihat_adsr, hihat_filter)
make_sound(bass_wave,  "bass",  bass_adsr,  bass_filter)
make_sound(pad_wave,   "pad",   pad_adsr,   pad_filter)

# ============================================
# 🧩 Assemblage simple des sons que tu as déjà
# (1 mesure à 128 BPM, 16 pas)
# ============================================
BPM = 128
steps_per_bar = 16
beat_dur = 60.0 / BPM            # durée d'une noire
step_dur = beat_dur / 4.0        # 1/16
bar_dur = 4 * beat_dur           # 1 mesure (≈ 1.875 s à 128 BPM)
N = int(bar_dur * fs)
mix = np.zeros(N, dtype=np.float32)

def render_clip(raw_wave, adsr, filt, length_s):
    """Prend un signal brut déjà généré (tes waves) et applique ADSR + filtre, puis tronque à length_s."""
    length = int(length_s * fs)
    w = raw_wave[:length].copy()
    env, _ = gen_adsr(
        attack=adsr.get("attack",0.01),
        decay=adsr.get("decay",0.1),
        sustain_level=adsr.get("sustain_level",0.7),
        sustain_time=max(0.0, length_s - adsr.get("attack",0.01) - adsr.get("decay",0.1) - adsr.get("release",0.2)),
        release=adsr.get("release",0.2),
        fs=fs
    )
    w = apply_envelope(w, env)
    if filt is not None:
        if filt.get("type") == "bandpass":
            b, a = design_filter("bandpass", band=filt["band"], order=filt.get("order",2), fs=fs)
        else:
            b, a = design_filter(filt["type"], cutoff=filt["cutoff"], order=filt.get("order",2), fs=fs)
        w = apply_filter(w, b, a)
    return w.astype(np.float32)

def drop(clip, at_time_s):
    i = int(at_time_s * fs)
    j = min(i + len(clip), len(mix))
    if j > i:
        mix[i:j] += clip[:j-i]

# --- Longueurs très simples par élément ---
kick_len  = 0.18
hat_len   = 0.06
bass_len  = beat_dur/2           # croche
pad_len   = bar_dur              # toute la mesure

# --- Prépare les "one-shots" depuis tes sons existants ---
kick_hit  = render_clip(kick_wave,  kick_adsr,  kick_filter,  kick_len)
hat_hit   = render_clip(hihat_wave, hihat_adsr, hihat_filter, hat_len)
bass_hit  = render_clip(bass_wave,  bass_adsr,  bass_filter,  bass_len)
pad_hold  = render_clip(pad_wave,   pad_adsr,   pad_filter,   pad_len)

# --- Pattern minimal : 4/4, hats off-beat, basse en croches, pad tenu ---
# Kick sur 0, 4, 8, 12 (16e)
for s in [0, 4, 8, 12]:
    drop(kick_hit, s * step_dur)

# Hi-hat sur 2, 6, 10, 14
for s in [2, 6, 10, 14]:
    drop(hat_hit, s * step_dur)

# Basse sur les croches (0, 2, 4, 6, 8, 10, 12, 14)
for s in [0, 2, 4, 6, 8, 10, 12, 14]:
    drop(bass_hit, s * step_dur)

# Pad en continu
drop(pad_hold, 0.0)

# --- Normalisation simple & export ---
mix = normalize_audio(mix)
save_wav("output/techno_mix_simple.wav", mix, fs)
print("✅ Mix simple exporté : output/techno_mix_simple.wav (1 mesure, 128 BPM)")


