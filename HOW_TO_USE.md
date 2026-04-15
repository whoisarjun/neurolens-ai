# How To Use Neurolens AI

This document describes the current workflow in this repository.

It replaces the old README-style setup notes and reflects the code as it exists now.

## What The Pipeline Does

The training pipeline in `main.py` runs these stages for each dataset split:

1. Load split metadata from `data_jsons/train.json`, `data_jsons/val.json`, and `data_jsons/test.json`
2. Normalize and denoise raw audio into `CLEANED_DATA/`
3. Augment training audio with 3 additional variants per sample
4. Transcribe audio with Whisper
5. Extract:
   - 52 acoustic features
   - 29 linguistic features
   - 18 semantic features scored by a local LLM
6. Generate 1024-d HuBERT embeddings
7. Concatenate everything into a `1123`-dim feature vector
8. Fit a scaler on train only
9. Train a multitask model for:
   - MMSE regression
   - cognitive-status classification (`HC`, `MCI`, `AD`)
10. Save weights, scaler, and feature arrays

## Environment Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

The Mandarin model is optional in principle, but needed if you want richer `zh` linguistic features.

## External Runtime Requirements

### 1. Ollama

Semantic features are generated locally through Ollama.

Install and run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull ministral-3:8b
```

The current semantic scorer default is set in `features/semantics.py`.

### 2. Hugging Face Access

The repo expects a `.env` file with:

```env
HUGGINGFACE_TOKEN=...
```

This is used for model downloads in the Hugging Face stack.

## Expected Local Layout

The code assumes these local directories exist:

```text
DATA/            raw dataset audio
CLEANED_DATA/    normalized + denoised audio
cache/           cached transcripts, features, embeddings
data_jsons/      split metadata
models/          saved weights and scaler
```

Important:
- this repo is not self-contained without the local datasets
- many expensive pipeline stages rely on cache reuse
- `.gitignore` intentionally excludes datasets, caches, and generated feature arrays

## Split Metadata Format

Each item in `data_jsons/*.json` looks like:

```json
{
  "question": "Please describe everything you see happening in the picture.",
  "input": "DATA/SOME_DATASET/train/example.wav",
  "output": "CLEANED_DATA/SOME_DATASET/train/example.wav",
  "mmse": 24.0,
  "diagnosis": "MCI",
  "language": "en"
}
```

Notes:
- `mmse` may be missing for some samples
- `diagnosis` may be missing for some samples
- language defaults to `en` if omitted
- supported language aliases are normalized in `utils/language.py`

## Run Training

From the project root:

```bash
python main.py
```

During execution, the script will ask whether to reuse cache for:
- transcripts
- acoustic features
- linguistic features
- LLM semantic features
- audio embeddings

If you choose cache reuse, files are loaded from `cache/`.

## Outputs Produced By Training

### Saved weights

```text
models/model_weights_reg.pth
models/model_weights_cls.pth
models/model_scaler.pkl
```

### Saved feature arrays

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
models/features/y_*_mask.npy
models/features/z_*_mask.npy
```

## Current Feature Breakdown

Feature inventory:
- acoustic features: `52`
- linguistic features: `29`
- semantic features: `18`
- HuBERT embeddings: `1024`

Total: `1123`

Detailed descriptions are in `features/FEATURES.md`.

## Language Support

The current pipeline supports:
- English: `en`
- Mandarin: `zh`

Mandarin-specific handling includes:
- alternate tokenization and POS logic
- multilingual coherence embeddings
- pinyin-aware syllable logic
- language-specific semantic rubrics

Current practical limitation:
- training data includes both English and Mandarin
- validation and test splits in the current local workspace are English-only

## Core Modules

### Audio cleanup

`processing/cleanup.py`

Handles:
- resampling to 16 kHz
- mono conversion
- peak normalization
- noise reduction

### Transcription and embeddings

`processing/transcriber.py`

Handles:
- Whisper ASR
- filler counting
- optional Mandarin pinyin transcript field
- HuBERT embedding generation
- cache-backed reuse

### Feature extraction

`features/acoustics.py`
- pitch
- energy
- speaking rate
- pauses
- MFCCs
- spectral measures
- voice quality

`features/linguistics.py`
- lexical richness
- repetition/disfluency
- coherence
- syntactic complexity
- POS ratios
- idea density
- concreteness
- discourse drift

`features/semantics.py`
- LLM-based discourse and clinical scoring through Ollama

### Model

`ml/model.py`

Defines:
- modality-specific encoders
- fused shared backbone
- MMSE regression head
- cognitive classification head

## Evaluation Scripts

### General evaluation

```bash
python eval_scripts/quick_eval.py
python eval_scripts/dataset_breakdown.py
python eval_scripts/error_distribution_analysis.py
python eval_scripts/cross_lang_eval.py
```

### Training-analysis scripts

```bash
python eval_scripts/lambda_grid_search.py
python eval_scripts/modality_config_comparison.py
python eval_scripts/best_worst_predictions_analysis.py
```

These scripts expect the saved feature arrays and model weights to already exist.

## Semantic LLM Evaluation

The `llm_eval/` folder is for benchmarking the semantic-feature scoring process itself.

It contains scripts for:
- comparing different local LLMs
- measuring intra-LLM consistency
- measuring inter-LLM agreement
- comparing LLM outputs against human raters

Important:
- some of these scripts are experimental and not all are cleanly runnable as-is
- `llm_eval/run_eval.py` currently contains a broken function call typo
- `llm_eval/human_benchmarking.py` has import/path assumptions that may need cleanup before use

## Data Splitting

`data_jsons/data_shuffler.py` rebuilds train/val/test splits while grouping by subject ID to reduce leakage across splits.

That matters especially for datasets where multiple recordings exist per subject.

## Common Practical Notes

- Use the project virtualenv if your system `python3` does not have the required packages.
- The first full run is expensive because Whisper, HuBERT, and semantic scoring all need model downloads and compute.
- GPU is used automatically when available.
- Cache reuse matters a lot for iteration speed.
- Training augmentation is done by generating additional cleaned audio files, not by on-the-fly tensor augmentation during the main pipeline.

## Minimal Workflow

If you already have the datasets and environment ready:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
ollama serve
ollama pull ministral-3:8b
python main.py
python eval_scripts/quick_eval.py
```

## Credits

Concreteness resources used in this repository:

Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014).  
Concreteness ratings for 40 thousand generally known English word lemmas.  
Behavior Research Methods, 46(3), 904-911.

Xu, X., & Li, J. (2020).  
Concreteness/abstractness ratings for two-character Chinese words in MELD-SCH.  
PLoS ONE, 15(6), e0232133.  
https://doi.org/10.1371/journal.pone.0232133
