import os
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
from scipy.io import wavfile
from dotenv import load_dotenv

load_dotenv()

# normalize audio to 16 kHz mono wav
def normalize(from_fp: Path, to_fp: Path, verbose=False):
    # load audio
    if verbose:
        print('[NORM] Loading audio')
    y, sr = librosa.load(from_fp, sr=None, mono=False)

    # convert to mono
    if y.ndim == 2:
        y = y.mean(axis=0)

    # resample to 16 kHz
    if verbose:
        print('[NORM] Resampling audio')
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # ensure output directory exists
    to_fp.parent.mkdir(parents=True, exist_ok=True)

    # write as 16 kHz mono wav
    if verbose:
        print('[NORM] Writing new audio')
    sf.write(to_fp, y, target_sr)
    if verbose:
        print('[NORM] Done normalizing audio!')

# remove bg noise
def denoise(fp: Path, verbose=False):
    # load audio
    if verbose:
        print('[DN] Loading audio')
    rate, data = wavfile.read(str(fp))

    # noise reduction
    if verbose:
        print('[DN] Reducing noise')
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    if verbose:
        print('[DN] Writing new audio')
    wavfile.write(str(fp), rate, reduced_noise)
    if verbose:
        print('[DN] Done denoising!')
