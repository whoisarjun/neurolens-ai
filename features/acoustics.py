# Extraction of acoustic features

import re
import librosa
import webrtcvad
import parselmouth
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import medfilt
from parselmouth.praat import call

# vad (voice activity detection)
def _vad(y: np.ndarray, sr: int):
    # normalize everything to max int
    peak = np.max(np.abs(y)) if y.size > 0 else 0.0
    if peak > 0:
        y_norm = y / peak
    else:
        y_norm = y

    pcm = (y_norm * 32767).astype(np.int16)

    # slice audio into 30 ms frames
    vad = webrtcvad.Vad(3)
    frame_len = int(sr * 0.03)
    num_frames = len(pcm) // frame_len
    pcm = pcm[:num_frames * frame_len]

    voiced_chunks = []

    pause_count = 0
    in_pause = False
    current_pause_frames = 0

    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        frame = pcm[start:end]
        frame_bytes = frame.tobytes()

        is_speech = vad.is_speech(frame_bytes, sr)

        if is_speech:
            voiced_chunks.append(frame)
            if in_pause and current_pause_frames >= 8:
                pause_count += 1
            in_pause = False
            current_pause_frames = 0
        else:
            in_pause = True
            current_pause_frames += 1

    # After the for loop, add a final check to count a trailing pause
    if in_pause and current_pause_frames >= 8:
        pause_count += 1

    if not voiced_chunks:
        # no speech detected, return empty signal and zero durations/pauses
        return np.array([], dtype=np.float32), sr, 0.0, 0

    # stitch voiced frames back together
    voiced_pcm = np.concatenate(voiced_chunks)
    y_vad = voiced_pcm.astype(np.float32) / 32767.0

    return y_vad, sr, len(y_vad) / sr, pause_count

# sound and point process (glottal pulses)
def _parselmouth_args(y, sr):
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch_min, pitch_max = 75, 500  # reasonable mix/max human speech pitch range
    point_process = call(snd, "To PointProcess (periodic, cc)", pitch_min, pitch_max)  # get glottal pulses

    return snd, point_process

# ========== FEATURE EXTRACTION ========== #

# Features 1-5, 11: F0 (Pitch), Pause count
def _f0(fn: Path):
    # load audio
    y, sr = librosa.load(str(fn), sr=16000, mono=True)
    duration = len(y) / sr

    y, sr, vad_duration, pause_count = _vad(y, sr)

    # use yin function to estimate f0
    f0 = librosa.yin(
        y,
        fmin=70,
        fmax=350,
        sr=sr
    )

    # remove silent/broken frames
    f0 = medfilt(f0, kernel_size=5)
    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) == 0:  # just in case
        mean_f0 = 0.0
        std_f0 = 0.0
        min_f0 = 0.0
        max_f0 = 0.0
    else:
        mean_f0 = float(np.mean(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        min_f0 = float(np.min(voiced_f0))
        max_f0 = float(np.max(voiced_f0))

    if len(voiced_f0) > 0:
        q25, q75 = np.percentile(voiced_f0, [25, 75])
        f0_iqr = float(q75 - q25)
    else:
        f0_iqr = 0.0

    return mean_f0, std_f0, min_f0, max_f0, f0_iqr, duration, vad_duration, pause_count, y, sr

# Features 6-8: Energy
def _energy(fn: Path):
    # load audio
    y, sr = librosa.load(str(fn), sr=16000, mono=True)

    # frame-wise rms energy
    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=512
    )[0]

    # compute energy stats
    mean_energy = float(np.mean(rms))
    std_energy = float(np.std(rms))
    min_energy = float(np.min(rms))
    max_energy = float(np.max(rms))
    dynamic_range = float(max_energy - min_energy)

    return mean_energy, std_energy, dynamic_range

# Features 9-10: Speed
def _speed(transcript: dict, duration: float):
    def _count_syllables(word: str):
        word = word.lower()
        word = re.sub('[^a-z]', '', word)

        if len(word) == 0:
            return 0

        # find vowel groups
        vowel_groups = re.findall(r'[aeiouy]+', word)
        syllables = len(vowel_groups)

        if word.endswith('e') and len(vowel_groups) > 1:
            syllables -= 1

        return max(1, syllables)
    words = re.findall(r'\b\w+\b', transcript.get('text'))
    words_ps = len(words) / duration
    syllables_ps = sum([_count_syllables(w) for w in words]) / duration

    return words_ps, syllables_ps

# Features 12-13: Pauses
def _pauses(duration: float, vad_duration: float):
    total_pause = duration - vad_duration
    pause_ratio = total_pause / duration
    return total_pause, pause_ratio

# Features 14-39: MFCCs
def _mfccs(y, sr):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    mfcc_features = []

    for i in range(len(mfcc_means)):
        mfcc_features.append(mfcc_means[i])
        mfcc_features.append(mfcc_stds[i])

    return mfcc_features

# Features 40-46: Spectral features
def _spectral(y, sr):
    S = np.abs(librosa.stft(y))

    centroid = librosa.feature.spectral_centroid(S=S)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S)[0]

    flux = librosa.onset.onset_strength(y=y, sr=sr)
    flux_mean = float(flux.mean())
    flux_std = float(flux.std())

    spec = S.mean(axis=1)
    freqs = np.arange(len(spec))
    spectral_slope = float(np.polyfit(freqs, spec, 1)[0])

    return float(np.mean(centroid)), float(np.std(centroid)), float(np.mean(bandwidth)), float(np.std(bandwidth)), flux_mean, flux_std, spectral_slope

# Features 47-52: Voice quality
def _voice_quality(y, sr):
    snd, pp = _parselmouth_args(y, sr)

    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.5)
    hnr_mean = call(harmonicity, "Get mean", 0, 0)

    pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpp = call(pc, 'Get CPPS', 'no', 0.01, 0.001, 60, 330, 0.05, 'parabolic', 0.001, 0.0, 'Straight', 'Robust')

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(zcr.mean())
    zcr_std = float(zcr.std())

    return jitter, shimmer, hnr_mean, cpp, zcr_mean, zcr_std

# ========== COMBINE EVERYTHING ========== #

def extract(fn: Path, transcript: dict, verbose=False):
    if verbose:
        print('[ACOU] Extracting acoustic features')

    mean_f0, std_f0, min_f0, max_f0, f0_iqr, duration, vad_duration, pause_count, vad_y, vad_sr = _f0(fn)
    mean_energy, std_energy, energy_range = _energy(fn)
    words_ps, syllables_ps = _speed(transcript, vad_duration)
    total_pauses, pause_ratio = _pauses(duration, vad_duration)
    mfccs = _mfccs(vad_y, vad_sr)
    spectral_centroid_mean, spectral_centroid_std, spectral_bandwidth_mean, spectral_bandwidth_std, spectral_flux_mean, spectral_flux_std, spectral_slope = _spectral(vad_y, vad_sr)
    jitter, shimmer, hnr, cpp, zcr_mean, zcr_std = _voice_quality(vad_y, vad_sr)

    ACOUSTIC_FEATURES = np.array([
        # F0/Pitch (5)
        mean_f0, std_f0, min_f0, max_f0, f0_iqr,

        # Energy (3)
        mean_energy, std_energy, energy_range,

        # Speaking rate (2)
        words_ps, syllables_ps,

        # Pauses (3)
        pause_count, total_pauses, pause_ratio,

        # MFCC means and stds (26)
        *mfccs,

        # Spectral (7)
        spectral_centroid_mean, spectral_centroid_std,
        spectral_bandwidth_mean, spectral_bandwidth_std,
        spectral_flux_mean, spectral_flux_std,
        spectral_slope,

        # Voice quality (6)
        jitter, shimmer, hnr, cpp,
        zcr_mean, zcr_std
    ])

    if verbose:
        print('[ACOU] Done extracting')

    return ACOUSTIC_FEATURES
