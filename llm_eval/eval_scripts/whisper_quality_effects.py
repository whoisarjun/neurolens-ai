import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import whisper
from scipy.stats import pearsonr
from tqdm import tqdm

AUDIO_DIR = Path("EVAL_DATA")
FILES = ['deepseek', 'granite', 'llama', 'ministral', 'qwen']
NUM_FEATURES = 18

whisper_model = whisper.load_model("large-v3-turbo")

def confidence(fp: Path):
    result = whisper_model.transcribe(
        str(fp),
        language='en',
        task='transcribe',
        temperature=0.0,
        best_of=1,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt='The sentence may be cut off, do not make up words to fill in the rest of the sentence. Um, like, you know, uh, so, basically',
        no_speech_threshold=0.6,
        logprob_threshold=1.0,
        compression_ratio_threshold=1.8,
        word_timestamps=False
    )

    total_tokens = sum(len(seg['tokens']) for seg in result['segments'])
    weighted_logprob = sum(
        seg['avg_logprob'] * len(seg['tokens'])
        for seg in result['segments']
    ) / total_tokens

    return np.exp(weighted_logprob)


def feature_variances(result_block):
    variances = []
    for i in range(NUM_FEATURES):
        vals = [run[i] for run in result_block]
        variances.append(max(vals) - min(vals))
    return variances

audio_paths = sorted([p for p in AUDIO_DIR.glob("*.wav")])

CACHE_DIR = Path("cache/whisper")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "conf_len.pkl"

if CACHE_FILE.exists():
    print(f"Loading Whisper cache from {CACHE_FILE}...")
    with CACHE_FILE.open("rb") as f:
        cache = pickle.load(f)

    cached_paths = cache.get("audio_paths")
    if cached_paths == [str(p) for p in audio_paths]:
        whisper_scores = np.array(cache["whisper_scores"])
        audio_lengths = np.array(cache["audio_lengths"])
        print("Cache valid. Using cached Whisper confidences and durations.")
    else:
        print("Cache mismatch with current audio set. Recomputing...")
        whisper_scores = []
        audio_lengths = []

        for fp in tqdm(audio_paths):
            whisper_scores.append(confidence(fp))
            duration = librosa.get_duration(path=str(fp))
            audio_lengths.append(duration)

        whisper_scores = np.array(whisper_scores)
        audio_lengths = np.array(audio_lengths)

        cache = {
            "audio_paths": [str(p) for p in audio_paths],
            "whisper_scores": whisper_scores.tolist(),
            "audio_lengths": audio_lengths.tolist(),
        }
        with CACHE_FILE.open("wb") as f:
            pickle.dump(cache, f)
        print(f"Updated cache written to {CACHE_FILE}.")
else:
    print("Computing Whisper confidences + durations (no cache found)...")
    whisper_scores = []
    audio_lengths = []

    for fp in tqdm(audio_paths):
        whisper_scores.append(confidence(fp))
        duration = librosa.get_duration(path=str(fp))
        audio_lengths.append(duration)

    whisper_scores = np.array(whisper_scores)
    audio_lengths = np.array(audio_lengths)

    cache = {
        "audio_paths": [str(p) for p in audio_paths],
        "whisper_scores": whisper_scores.tolist(),
        "audio_lengths": audio_lengths.tolist(),
    }
    with CACHE_FILE.open("wb") as f:
        pickle.dump(cache, f)
    print(f"Cache written to {CACHE_FILE}.")

for f in FILES:
    print(f"\n=============================")
    print(f"   LLM: {f.upper()}")
    print("=============================")

    with Path(f"models/features/{f}.json").open() as j:
        raw = json.load(j).get("results")
        results = [entry["results"] for entry in raw]

    all_variances = np.array([feature_variances(r) for r in results])

    # A. WHISPER CONFIDENCE VS VARIANCE

    print("\nA. Whisper Confidence vs Variance")
    print("| Feature | r(whisper_conf, variance) |")

    for i in range(NUM_FEATURES):
        r, _ = pearsonr(whisper_scores, all_variances[:, i])
        print(f"| {i:02d} | {r:.4f} |")

    overall_var = np.mean(all_variances, axis=1)
    r_overall, _ = pearsonr(whisper_scores, overall_var)
    print(f"| OVERALL | {r_overall:.4f} |")

    # B. AUDIO LENGTH VS VARIANCE

    print("\nB. Audio Duration vs Variance")
    print("| Feature | r(duration, variance) |")

    for i in range(NUM_FEATURES):
        r, _ = pearsonr(audio_lengths, all_variances[:, i])
        print(f"| {i:02d} | {r:.4f} |")

    r_overall2, _ = pearsonr(audio_lengths, overall_var)
    print(f"| OVERALL | {r_overall2:.4f} |")
