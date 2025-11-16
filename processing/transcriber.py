# Whisper-based ASR transcription

import librosa
import whisper
import warnings
from pathlib import Path

# Ignore this specific memory warning
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

print('[ASR] Loading whisper model')
model = whisper.load_model('large')

def asr(fp: Path, verbose=False):
    if verbose:
        print('[ASR] Transcribing')
    result = model.transcribe(str(fp))
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
        } for s in result.get('segments')]
    }
