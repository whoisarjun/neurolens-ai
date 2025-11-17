# Whisper-based ASR transcription

import torch
import librosa
import whisper
import warnings
import whisperx
from pathlib import Path
from faster_whisper import WhisperModel

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
modelx = whisperx.load_model(faster_whisper_model, device=DEVICE, compute_type=compute_type)

def asr(fp: Path, verbose=False):
    if verbose:
        print('[ASR] Transcribing')
    result = model.transcribe(str(fp),
                              condition_on_previous_text=False)
    if verbose:
        print('[ASR] Transcribing (CrisperWhisper)')
    resultx = modelx.transcribe(str(fp))
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
