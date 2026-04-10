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
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from utils import cache
from utils.language import normalize_language

try:
    from pypinyin import lazy_pinyin
except Exception:  # pragma: no cover - optional dependency
    lazy_pinyin = None

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

EN_INITIAL_PROMPT = (
    'The sentence may be cut off, do not make up words to fill in the rest '
    'of the sentence. Um, like, you know, uh, so, basically'
)
ZH_INITIAL_PROMPT = (
    '句子可能被截断，不要补全不存在的词语。'
    '保留原话中的停顿词，如 嗯、呃、那个、这个、就是、然后。'
)

EN_FILLER_PATTERN = re.compile(
    r'\b(um+|uh+|er+|ah+|like|you know|so|actually|basically|literally)\b',
    re.IGNORECASE
)
ZH_FILLER_PATTERN = re.compile(r'(嗯+|呃+|额+|啊+|那个|这个|就是|然后)')

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

def _count_fillers(text: str, language: str) -> int:
    if language == 'zh':
        return len(ZH_FILLER_PATTERN.findall(text))
    return len(EN_FILLER_PATTERN.findall(text.lower()))

def _text_to_pinyin(text: str) -> str | None:
    if lazy_pinyin is None:
        return None
    if not text:
        return ''
    try:
        return ' '.join(lazy_pinyin(text))
    except Exception:
        return None

def asr(fp: Path, use_cache=False, verbose=False, language='en'):
    model = get_whisper()
    lang = normalize_language(language)

    transcript = None
    cache_variant = None if lang == 'en' else f'asr:{lang}'
    cache_file = cache.key(fp, ASR_CACHE_DIR, variant=cache_variant)
    if use_cache:
        transcript = cache.load(cache_file)
    if transcript is None:
        if verbose:
            print('[ASR] Transcribing (Whisper)')

        result = model.transcribe(
            str(fp),
            language=lang,
            task='transcribe',
            temperature=0.0,
            best_of=1,
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=ZH_INITIAL_PROMPT if lang == 'zh' else EN_INITIAL_PROMPT,
            no_speech_threshold=0.6,
            logprob_threshold=1.0,
            compression_ratio_threshold=1.8
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
            'fillers': _count_fillers(result.get('text', ''), lang),
            'language': lang,
        }
        if lang == 'zh':
            transcript['text_pinyin'] = _text_to_pinyin(result.get('text', ''))
        cache.save(cache_file, transcript)
    elif transcript.get('language') is None:
        transcript['language'] = lang
    return transcript

def embeddings(file_paths: list[Path], use_cache=False, batch_size=16, tqdm_desc=''):
    hubert_model, hubert_processor = get_hubert()

    results = {}
    to_process = []

    # check cache first
    for fp in file_paths:
        cache_file = cache.key(fp, EMB_CACHE_DIR)
        if use_cache:
            emb = cache.load(cache_file)
            if emb is not None:
                results[fp] = emb.cpu()
                continue
        y, sr = librosa.load(str(fp), sr=16000, mono=True)
        to_process.append((fp, y, len(y) / sr))

    # Group similar durations together so padding overhead is lower.
    to_process.sort(key=lambda item: item[2])

    def run_batch(batch_items):
        batch_fps = [fp for fp, _, _ in batch_items]
        batch_audios = [y for _, y, _ in batch_items]
        max_len = max(len(y) for y in batch_audios)
        padded = [np.pad(y, (0, max_len - len(y))) for y in batch_audios]

        inputs = hubert_processor(
            padded,
            sampling_rate=16000,
            return_tensors='pt',
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = hubert_model(**inputs)
                embs = outputs.last_hidden_state.mean(dim=1)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch_items) == 1:
                fp, _, duration = batch_items[0]
                raise RuntimeError(
                    f'HuBERT OOM for {fp.name} ({duration:.1f}s). '
                    'Reduce audio length, free GPU memory, or force CPU execution.'
                ) from None

            mid = max(1, len(batch_items) // 2)
            run_batch(batch_items[:mid])
            run_batch(batch_items[mid:])
            return
        finally:
            del inputs
            if 'outputs' in locals():
                del outputs

        for fp, emb in zip(batch_fps, embs):
            emb_cpu = emb.unsqueeze(0).cpu()
            results[fp] = emb_cpu
            cache.save(cache.key(fp, EMB_CACHE_DIR), emb_cpu)

        del embs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # process uncached files in batches (default 16)
    for i in tqdm(range(0, len(to_process), batch_size), desc=tqdm_desc):
        run_batch(to_process[i:i + batch_size])

    return results

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
