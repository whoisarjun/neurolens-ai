# Whisper-based ASR transcription

import hashlib
import pickle
import tempfile
import warnings
from pathlib import Path

import librosa
import torch
import whisper
import whisperx
import numpy as np
import soundfile as sf
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Ignore this specific memory warning
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32"
faster_whisper_model = 'nyrahealth/faster_CrisperWhisper'

CACHE_DIR = Path('cache/embeddings')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print('[ASR] Loading whisper model')
model = whisper.load_model('large-v3-turbo')
print('[ASR] Loading crisper-whisper model')
modelx = whisperx.load_model(faster_whisper_model, device=DEVICE, compute_type=compute_type, vad_method='silero')

print('[EMB] Loading hubert-large-ll60k')
hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
hubert_model.eval()


def _get_cache_key(filename: Path):
    key = str(filename).split('DATA/')[-1]
    return hashlib.md5(key.encode()).hexdigest()


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

def embeddings(fp: Path, verbose=False):
    # check cache if embeddings have been saved
    cache_key = _get_cache_key(fp)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        if verbose:
            print('[HUBERT] Loading cached')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if verbose:
        print('[HUBERT] Computing embeddings')

    # load audio and process
    y, sr = librosa.load(str(fp), sr=16000, mono=True)
    inputs = hubert_processor(y, sampling_rate=16000, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert_model(**inputs)
        # mean pool over time dimension
        emb = outputs.last_hidden_state.mean(dim=1)

    # cache it
    with open(cache_file, 'wb') as f:
        pickle.dump(emb.cpu(), f)

    return emb.cpu()

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