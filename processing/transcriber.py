# Whisper-based ASR transcription

import torch
import librosa
import whisper
import tempfile
import warnings
import whisperx
import numpy as np
import soundfile as sf
from pathlib import Path

# Ignore this specific memory warning
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32"
faster_whisper_model = 'nyrahealth/faster_CrisperWhisper'

print('[ASR] Loading whisper model')
model = whisper.load_model('large-v3-turbo')
print('[ASR] Loading crisper-whisper model')
modelx = whisperx.load_model(faster_whisper_model, device=DEVICE, compute_type=compute_type, vad_method='silero')

def normalize_audio(fp: Path):
    try:
        y, sr = librosa.load(str(fp), sr=16000)

        peak = np.max(np.abs(y))
        if peak < 1e-6:
            return fp

        # peak normalize
        y = y / peak

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, y, sr)
        return Path(tmp.name)

    except Exception as e:
        print(f'[WARN] Normalization failed: {e}')
        return fp


def asr(fp: Path, verbose=False):
    if verbose:
        print('[ASR] Transcribing')
    result = model.transcribe(
        str(fp),
        language='en',
        task='transcribe',
        temperature=0.0,
        best_of=1,
        beam_size=5,
        condition_on_previous_text=False
    )

    if verbose:
        print('[ASR] Transcribing (CrisperWhisper)')
    norm_fp = normalize_audio(fp)
    try:
        resultx = modelx.transcribe(str(norm_fp), language='en')
    finally:
        if norm_fp != fp:
            norm_fp.unlink(missing_ok=True)

    if verbose:
        print('[ASR] Done transcribing')

    y, sr = librosa.load(str(fp), sr=16000)
    return {
        'text': result.get('text'),
        'duration': len(y) / sr,
        'segments': [{
            'text': s.get('text'),
            'start': s.get('start'),
            'end': s.get('end')
        } for s in result.get('segments')],
        'filler_count': sum([s.get('text').count('[') for s in resultx.get('segments')])
    }

def sanitize_text(text: str, max_chars: int = 8000, max_repeat: int = 50) -> str:
    tokens = text.split()
    cleaned = []
    last = None
    count = 0
    for t in tokens:
        if t == last:
            count += 1
            if count <= max_repeat:
                cleaned.append(t)
        else:
            last = t
            count = 1
            cleaned.append(t)
    s = " ".join(cleaned)
    return s[:max_chars]
