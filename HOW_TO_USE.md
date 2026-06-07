# Using Neurolens AI

This guide describes the repository as currently implemented. It distinguishes
the new PCA-based training pipeline from older tracked weights, scalers, plots,
and evaluation outputs.

## Prerequisites

- Python 3.10 or newer
- local access to the datasets referenced by `data_jsons/*.json`
- Ollama running locally
- sufficient disk space for raw audio, cleaned audio, three training
  augmentations per sample, model downloads, and caches
- optional CUDA-capable GPU; PyTorch uses CUDA automatically when available

The repository does not include or download the clinical speech corpora.
Flask is also not listed in `requirements.txt`; it is only needed when working
on the experimental API.

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the spaCy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

The extractors can fall back to blank or heuristic language pipelines when a
spaCy model is unavailable, but the resulting features are not equivalent.

Install and prepare Ollama:

```bash
ollama pull ministral-3:8b
ollama serve
```

`features/semantics.py` communicates with the local Ollama service and uses
`ministral-3:8b` by default.

Hugging Face and sentence-transformer models are downloaded on first use. The
current source does not explicitly read `HUGGINGFACE_TOKEN`; standard Hugging
Face environment variables or prior CLI authentication can be used if a
download requires authentication.

## Local Directory Layout

Run commands from the repository root. Expected paths are:

```text
DATA/                    raw recordings referenced by split metadata
CLEANED_DATA/            normalized, denoised, and augmented WAV files
cache/
  transcripts/
  acoustics/
  linguistics/
  semantics/
  embeddings/
data_jsons/
  train.json
  val.json
  test.json
models/
  features/              generated NumPy arrays
```

`DATA/`, `CLEANED_DATA/`, `cache/`, `data_jsons/`, and generated feature arrays
are ignored by Git. The current workspace contains local split metadata, but a
fresh clone will not receive it from Git.

## Split Metadata

Each split file must contain a top-level `data` array:

```json
{
  "data": [
    {
      "question": "Describe everything happening in the picture.",
      "input": "DATA/CORPUS/example.wav",
      "output": "CLEANED_DATA/CORPUS/example.wav",
      "mmse": 24,
      "diagnosis": "MCI",
      "language": "en"
    }
  ]
}
```

Field behavior:

| Field | Required | Notes |
|---|---|---|
| `question` | yes | Used by the semantic scorer |
| `input` | yes | Source recording |
| `output` | yes | Cleaned 16 kHz mono WAV |
| `mmse` | no | Numeric when present; missing values are masked |
| `diagnosis` | no | One of `HC`, `MCI`, or `AD`; missing values are masked |
| `language` | no | Defaults to `en`; aliases normalize to `en` or `zh` |

Unsupported language strings currently fall back to the provided default,
usually English. They do not raise an error in the training pipeline.

## Training Pipeline

Run:

```bash
python main.py
```

At startup, the script asks whether to reuse each cache:

```text
Use cache for transcripts? (y/n):
Use cache for acoustic features? (y/n):
Use cache for linguistic features? (y/n):
Use cache for LLM-generated semantic features? (y/n):
Use cache for audio embeddings? (y/n):
```

Only lowercase `y` enables reuse.

For each split, `main.py` performs:

1. audio normalization and denoising;
2. three offline audio augmentations for the training split only;
3. Whisper transcription;
4. extraction of 52 acoustic, 29 linguistic, and 18 semantic features;
5. generation of 1024-dimensional HuBERT representations; and
6. target parsing with independent MMSE and diagnosis masks.

It then fits preprocessing on training data only:

1. `StandardScaler` on the 99 acoustic, linguistic, and semantic features;
2. a separate `StandardScaler` on the 1024 HuBERT dimensions; and
3. `PCA(n_components=128)` on the scaled training HuBERT representations.

The transformed blocks are concatenated:

```text
99 standardized features + 128 PCA components = 227 model inputs
```

Validation and test data are transformed with the training-fitted objects.
Training runs for 50 epochs with batch size 64 and saves the checkpoint with
the best validation score:

```text
score = -validation_MAE + 2 * validation_macro_F1
```

If only one validation target is available, the score uses the available
metric. Training fails if the validation split has neither target.

## Generated Artifacts

Current training writes:

```text
models/model_scaler.pkl          scaler for the 99 non-HuBERT features
models/model_emb_scaler.pkl      scaler for the 1024 HuBERT dimensions
models/model_emb_pca.pkl         HuBERT PCA projection, 1024 -> 128
models/model_weights_reg.pth     best MMSE regressor checkpoint
models/model_weights_cls.pth     best classification checkpoint
```

It also writes:

```text
models/features/X_train_scaled.npy
models/features/X_val_scaled.npy
models/features/X_test_scaled.npy
models/features/y_train.npy
models/features/y_val.npy
models/features/y_test.npy
models/features/z_train.npy
models/features/z_val.npy
models/features/z_test.npy
models/features/y_train_mask.npy
models/features/y_val_mask.npy
models/features/y_test_mask.npy
models/features/z_train_mask.npy
models/features/z_val_mask.npy
models/features/z_test_mask.npy
```

`X_*_scaled.npy` should have shape `(samples, 227)` when produced by the
current code. Training augmentation makes the generated training arrays four
times larger than the original training metadata.

## Cache Behavior

Caches are pickle files keyed primarily by the audio path. Mandarin
transcript, linguistic, and semantic caches include a language-specific
variant. Acoustic and HuBERT caches do not.

Important consequences:

- changing extraction code does not invalidate an existing cache;
- changing file contents without changing the path can leave stale cache data;
- English cache keys retain the original unversioned format;
- augmented recordings always reuse the base transcript;
- augmented recordings share the base semantic cache key, but recompute that
  score for every augmentation when semantic-cache reuse is disabled;
- selecting `n` recomputes a stage and overwrites its cache entry.

Delete the relevant cache directory manually when feature definitions, model
versions, prompts, or source audio have changed.

## Models Downloaded at Runtime

| Purpose | Model |
|---|---|
| ASR | Whisper `large-v3-turbo` |
| Speech representation | `facebook/hubert-large-ll60k` |
| English sentence similarity | `all-mpnet-base-v2` |
| Mandarin sentence similarity | `paraphrase-multilingual-mpnet-base-v2` |
| Semantic scoring | Ollama `ministral-3:8b` |

Whisper and HuBERT are lazily loaded and unloaded between major pipeline
stages. HuBERT batches recordings by duration and recursively splits a batch
after a CUDA out-of-memory error.

## Data Splitting

`data_jsons/data_shuffler.py` has two interactive modes:

```bash
python data_jsons/data_shuffler.py
```

- mode `1` prints current split statistics without modifying files;
- mode `2` rewrites all three split files.

Mode 2 groups recordings by corpus and inferred subject identifier, targets an
80/10/10 split, and heuristically balances corpus, language, diagnosis, and
language-diagnosis composition. Back up local metadata before using mode 2.

## Evaluation and Compatibility

The source tree is in a transition from an older 1123-dimensional model to the
current 227-dimensional PCA-based model.

| Component | Current status |
|---|---|
| `main.py` | Implements 99 + PCA-128 = 227 inputs |
| `ml/model.py` | Expects 227 inputs by default |
| tracked `models/model_scaler.pkl` | Historical; fitted on 1123 inputs |
| tracked `models/model_weights_*.pth` | Historical; expect 1024 HuBERT inputs |
| tracked `eval_results/*` | Historical; generated from older data/artifacts |
| `eval_scripts/quick_eval.py` | Structurally compatible after retraining |
| `eval_scripts/cross_lang_eval.py` | Structurally compatible after retraining |
| `eval_scripts/lambda_grid_search.py` | Structurally compatible after retraining |
| `eval_scripts/dataset_breakdown.py` | Does not apply missing-MMSE masks |
| `eval_scripts/error_distribution_analysis.py` | Does not apply missing-MMSE masks |
| `eval_scripts/best_worst_predictions_analysis.py` | Does not apply missing-MMSE masks |
| `eval_scripts/modality_config_comparison.py` | Stale: assumes 1024 saved embedding columns |
| `test.py` | Duplicate quick-evaluation utility |

After a successful current-code training run, the least stale evaluation entry
point is:

```bash
python eval_scripts/quick_eval.py
```

Do not interpret the tracked CSV metrics or plots as results for the current
architecture. Regenerate them with matching 227-dimensional arrays and
checkpoints first.

## Experimental Inference API

`server.py` exposes:

```text
GET  /health
POST /predict
```

`/predict` accepts exactly five uploaded audio files, five matching questions,
and one language (`en` or `zh`). It averages the five raw feature vectors and
returns named handcrafted/semantic features, the averaged 1024 HuBERT values,
an MMSE prediction, and a cognitive-status prediction.

However, the server currently implements the old preprocessing path:

- it expects a 1123-dimensional vector;
- it loads only `model_scaler.pkl`;
- it does not load `model_emb_scaler.pkl` or `model_emb_pca.pkl`; and
- it constructs the current 227-input model.

The response schema also labels acoustic indices 9 and 10 as
`syllables_per_sec` then `words_per_sec`, while the extractor emits words per
second then syllables per second. Temporary upload directories are not removed
because the cleanup call is currently commented out.

Therefore the inference server is not compatible with either the tracked old
artifacts or a newly trained current model without code changes. In addition,
`python server.py` calls `_load_inference_models()` before starting Flask, so
the incompatible tracked weights prevent even `/health` from becoming
reachable through the standard launch command. `/predict` should be considered
unavailable for valid current-model inference.

## Semantic-Rater Evaluation

`llm_eval/` contains exploratory utilities for:

- repeated scoring by several local LLMs;
- inter-model and intra-model agreement;
- comparison with human ratings; and
- analysis of ASR quality effects.

This directory is not part of the supported training path. In particular:

- `llm_eval/run_eval.py` calls a misspelled pipeline function;
- `llm_eval/human_benchmarking.py` uses working-directory-dependent imports
  and paths; and
- its data and generated model-output directories are intentionally ignored.

Treat these scripts as experimental notebooks-in-code rather than stable CLI
tools.

## Troubleshooting

### Ollama connection or model errors

Confirm the service is running and the configured model exists:

```bash
ollama list
ollama serve
```

Semantic parse failures are retried by the batch pipeline. Repeated failure
falls back to eighteen scores of `1.0`.

### Missing spaCy models

Install both language packages:

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

Without them, syntax and POS features use reduced fallbacks.

### HuBERT PCA failure

PCA with 128 components requires at least 128 augmented training rows and at
least 128 HuBERT dimensions. The latter is always satisfied; very small custom
training sets may not satisfy the sample requirement.

### Shape mismatch while loading weights

The tracked checkpoints use the earlier 1123-dimensional architecture.
Retrain with current code and ensure all of the following come from the same
run:

```text
model_scaler.pkl
model_emb_scaler.pkl
model_emb_pca.pkl
model_weights_reg.pth
model_weights_cls.pth
models/features/X_*_scaled.npy
```

### No speech detected

The acoustic extractor expects voiced frames. Silent or severely corrupted
recordings can fail downstream pitch, spectral, or voice-quality operations
despite numeric sanitization at the final feature-vector step.

## Data and Ethics

The corpora contain sensitive health-related speech. Follow each provider's
access agreement, attribution requirements, consent limitations, and data
security rules. Do not commit raw recordings, transcripts, cache files, or
participant metadata.

Neurolens AI is for research use only and must not be used as a standalone
diagnostic or clinical decision system.

## Corpus Credits

The project uses local data derived from ADReSS-IS2020, ADReSSo21, ADReSS-M,
the DementiaBank Pitt Corpus, the DementiaBank Mandarin Chou Corpus,
TAUKADIAL's English and Mandarin/Chinese data, and NCMMSC2021-AD. Canonical
references and provider links are listed in the
[README dataset acknowledgements](README.md#dataset-acknowledgements).
