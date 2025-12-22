# Whisper-based ASR transcription

import gc
import logging
import re
import tempfile
import warnings
from pathlib import Path

import librosa
import torch
import whisper
import numpy as np
import soundfile as sf
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from utils import cache

# Ignore this specific memory warning
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='torchaudio')
warnings.filterwarnings('ignore', module='speechbrain')
warnings.filterwarnings('ignore', module='pyannote')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32"

ASR_CACHE_DIR = Path('cache/transcripts')
EMB_CACHE_DIR = Path('cache/embeddings')
ASR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# None placeholders for lazy loading
model = None
hubert_processor = None
hubert_model = None

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

# Model loaders (only load when used)
def get_whisper():
    global model
    if model is None:
        print('[ASR] Loading whisper')
        model = whisper.load_model('large-v3-turbo')
    return model

def get_hubert():
    global hubert_model, hubert_processor
    if hubert_model is None:
        print('[EMB] Loading HuBERT')
        hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            'facebook/hubert-large-ll60k'
        )
        hubert_model = HubertModel.from_pretrained(
            'facebook/hubert-large-ll60k'
        )
        hubert_model = hubert_model.to(DEVICE)
        hubert_model.eval()
    return hubert_model, hubert_processor

# Model unloaders (when not in use)
def unload_models():
    global model, hubert_model, hubert_processor

    model = None
    hubert_model = None
    hubert_processor = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def asr(fp: Path, use_cache=False, verbose=False):
    model = get_whisper()

    transcript = None
    cache_file = cache.key(fp, ASR_CACHE_DIR)
    if use_cache:
        transcript = cache.load(cache_file)
    if transcript is None:
        if verbose:
            print('[ASR] Transcribing (Whisper)')

        result = model.transcribe(
            str(fp),
            language='en',
            task='transcribe',
            temperature=0.0,
            best_of=1,
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt='Um, like, you know, uh, so, basically'
        )

        if verbose:
            print('[ASR] Done transcribing (Whisper)')

        y, sr = librosa.load(str(fp), sr=16000)
        transcript = {
            'text': result.get('text'),
            'duration': len(y) / sr,
            'segments': [{
                'text': s.get('text'),
                'start': s.get('start'),
                'end': s.get('end')
            } for s in result.get('segments')],
            'fillers': len(re.findall(r'\b(um+|uh+|er+|ah+|like|you know|so|actually|basically|literally)\b', result.get('text').lower()))
        }
        cache.save(cache_file, transcript)
    return transcript

def embeddings(fp: Path, use_cache=False, verbose=False):
    hubert_model, hubert_processor = get_hubert()

    emb = None
    cache_file = cache.key(fp, EMB_CACHE_DIR)
    if use_cache:
        emb = cache.load(cache_file)
    if emb is None:
        # caculation time
        # load audio and process
        y, sr = librosa.load(str(fp), sr=16000, mono=True)
        inputs = hubert_processor(y, sampling_rate=16000, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = hubert_model(**inputs)
            # mean pool over time dimension
            emb = outputs.last_hidden_state.mean(dim=1)

        cache.save(cache_file, emb)

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