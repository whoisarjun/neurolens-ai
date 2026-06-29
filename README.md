# Neurolens AI

Multimodal speech analysis for cognitive-status classification and MMSE
regression.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20code-6c757d)
![Languages](https://img.shields.io/badge/languages-English%20%7C%20Mandarin-198754)

Neurolens AI investigates whether acoustic, linguistic, discourse-semantic,
and self-supervised speech representations can be combined to estimate
cognitive status from spontaneous speech. The current pipeline supports
English and Mandarin recordings and jointly predicts:

- Mini-Mental State Examination (MMSE) score
- cognitive status: healthy control (`HC`), mild cognitive impairment (`MCI`),
  or Alzheimer's disease (`AD`)

This repository is research code. It is not a clinical device, diagnostic
system, packaged Python library, or self-contained dataset release.

## Overview

For each recording, the pipeline:

1. converts audio to denoised 16 kHz mono WAV;
2. transcribes it with Whisper `large-v3-turbo`;
3. extracts 52 acoustic, 29 linguistic, and 18 LLM-derived semantic features;
4. computes a 1024-dimensional HuBERT `facebook/hubert-large-ll60k`
   representation;
5. standardizes the 99 handcrafted/semantic features separately from the
   HuBERT representation;
6. reduces HuBERT from 1024 to 128 dimensions using PCA fitted on training
   data only; and
7. trains a shared multitask network on the resulting 227-dimensional input.

```text
audio + elicitation question
        |
        +-- audio cleanup
        +-- Whisper transcription
        |
        +-- acoustic features -------------------- 52
        +-- linguistic features ----------------- 29
        +-- local-LLM semantic scores ----------- 18
        +-- HuBERT representation -- 1024 -> PCA 128
                                                     |
                                  fused model input: 227
                                                     |
                           +-------------------------+-------------------+
                           |                                             |
                    MMSE regression                         HC/MCI/AD classification
```

See [HOW_TO_USE.md](HOW_TO_USE.md) for setup and execution, and
[features/FEATURES.md](features/FEATURES.md) for the ordered feature schema.

## Method

### Speech processing

Audio is resampled to 16 kHz, converted to mono, peak-normalized, and
denoised. Training samples receive three offline augmentations using small
time-stretch perturbations, pink noise, and light reverberation. Augmented
audio always reuses the base recording's transcript. It shares the base
semantic cache key, so semantic scores are reused when that cache is enabled
and recomputed otherwise.

Whisper produces the transcript, segment timestamps, duration, filler count,
and language metadata. Mandarin transcripts can also include a pinyin
representation when `pypinyin` is available.

### Feature representation

| Modality | Raw dimensions | Model dimensions | Implementation |
|---|---:|---:|---|
| Acoustic | 52 | 52 | `features/acoustics.py` |
| Linguistic | 29 | 29 | `features/linguistics.py` |
| Semantic | 18 | 18 | `features/semantics.py` |
| HuBERT | 1024 | 128 after PCA | `processing/transcriber.py` |
| **Total** | **1123** | **227** | |

Semantic features are rubric-based scores in the range 0-4. They are produced
through a local Ollama model; the configured default is `ministral-3:8b`.
English and Mandarin use separate rubric files.

### Multitask model

The 227-dimensional input is partitioned by modality and passed through four
encoders:

- acoustics: `52 -> 64`
- linguistics: `29 -> 32`
- semantics: `18 -> 32`
- PCA-reduced HuBERT: `128 -> 64`

The encoded representations are concatenated into a 192-dimensional vector,
fused through a `192 -> 128 -> 64` shared backbone, and passed to separate
MMSE-regression and three-class classification heads. Training uses Huber loss for regression, cross-entropy
for classification, and masked losses so samples may omit either target.

The current training loss is:

```text
0.45 * MMSE loss + 0.55 * cognitive-status loss
```

when both labels are present.

## Data

The local, Git-ignored split metadata references seven corpora:

| Corpus key | Language in current /metadata | Primary labels used |
|---|---|---|
| `ADReSS-IS2020` | English | MMSE, diagnosis |
| `ADReSSo21` | English | MMSE, diagnosis |
| `ADReSS-M` | English | MMSE, diagnosis |
| `PITT_CORPUS` | English | MMSE, diagnosis |
| `TAUKADIAL` | English | MMSE, diagnosis |
| `CHOU` | Mandarin | diagnosis |
| `NCMMSC2021` | Mandarin | diagnosis |

The repository does not distribute the underlying recordings. Access,
licensing, consent, and citation requirements remain governed by each corpus
provider. The split builder groups recordings by subject and attempts to
balance split size, corpus, language, diagnosis, and language-diagnosis
composition.

The local TAUKADIAL subset is currently labelled English-only even though the
TAUKADIAL corpus itself is bilingual English and Mandarin/Chinese.

Current metadata contains 1,550 original recordings before training
augmentation:

| Split | Samples | English | Mandarin | MMSE-labelled | Diagnosis-labelled |
|---|---:|---:|---:|---:|---:|
| Train | 1,219 | 870 | 349 | 870 | 1,219 |
| Validation | 166 | 122 | 44 | 122 | 166 |
| Test | 165 | 122 | 43 | 122 | 165 |

These counts describe the local metadata currently present in
`data_jsons/`; they are not corpus-wide statistics.

## Quick Start

Prerequisites:

- Python 3.10 or newer
- system support for the audio packages in `requirements.txt`
- Ollama with `ministral-3:8b`
- local copies of the referenced datasets
- enough memory and storage for Whisper, HuBERT, sentence-transformer models,
  caches, cleaned audio, and augmented audio

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
ollama pull ministral-3:8b
ollama serve
python main.py
```

`main.py` interactively asks which cached stages to reuse. A first run may
download several model checkpoints and is computationally expensive.

## Outputs

A successful run of the current training code writes:

```text
models/
  model_scaler.pkl
  model_emb_scaler.pkl
  model_emb_pca.pkl
  model_weights_reg.pth
  model_weights_cls.pth
  features/
    X_{train,val,test}_scaled.npy
    y_{train,val,test}.npy
    z_{train,val,test}.npy
    y_{train,val,test}_mask.npy
    z_{train,val,test}_mask.npy
```

The saved `X_*_scaled.npy` arrays produced by current code have 227 columns.

## Evaluation Status

The scalers, PCA, and weights tracked in `models/`, along with the plots and
CSV files in `eval_results/`, correspond to the current PCA-based
227-dimensional architecture. The evaluation scripts in `eval_scripts/` run
against the 227-dimensional arrays written by `main.py`.

Regenerate the feature arrays, checkpoints, and evaluation artifacts from a
single training run before comparing metrics across changes. See
[HOW_TO_USE.md](HOW_TO_USE.md#evaluation-and-compatibility) for the per-script
evaluation notes.

## Repository Structure

```text
main.py                 end-to-end preprocessing and multitask training
features/               acoustic, linguistic, and semantic extraction
processing/             cleanup, ASR, embeddings, and batch orchestration
ml/                     augmentation and model implementation
data_jsons/             local train/validation/test metadata and split builder
eval_scripts/           evaluation and ablation utilities
eval_results/           generated plots and tables
llm_eval/               experimental semantic-rater evaluation utilities
models/                 tracked model artifacts; generated arrays ignored
```

## Known Limitations

- The project assumes locally licensed datasets and is not reproducible from
  the Git repository alone.
- The current model does not constrain MMSE predictions to the clinical
  0-30 range.
- Semantic scoring depends on a nondeterministic external runtime despite
  temperature-zero settings; failures can fall back to default scores.
- Sentence-transformer and spaCy model-loading failures can silently activate
  reduced linguistic fallbacks.
- Training does not set a global random seed, so augmentation and optimization
  are not exactly reproducible between runs.
- This system is intended for research only. Its outputs must not be treated
  as diagnoses or used for clinical decision-making.

## Dataset Acknowledgements

This project uses locally obtained data derived from the following corpora and
shared tasks. The data are not redistributed here.

- **ADReSS-IS2020 / ADReSS 2020**: Luz, S., Haider, F., de la Fuente, S.,
  Fromm, D., & MacWhinney, B. (2020). *Alzheimer's Dementia Recognition
  Through Spontaneous Speech: The ADReSS Challenge*. Interspeech 2020.
  [doi:10.21437/Interspeech.2020-2571](https://doi.org/10.21437/Interspeech.2020-2571)
- **ADReSSo21 / ADReSSo 2021**: Luz, S., Haider, F., de la Fuente, S.,
  Fromm, D., & MacWhinney, B. (2021). *Detecting Cognitive Decline Using
  Speech Only: The ADReSSo Challenge*. Interspeech 2021.
  [doi:10.21437/Interspeech.2021-1220](https://doi.org/10.21437/Interspeech.2021-1220)
- **ADReSS-M**: Luz, S., Haider, F., Fromm, D., Lazarou, I.,
  Kompatsiaris, I., & MacWhinney, B. (2023). *Multilingual Alzheimer's
  Dementia Recognition Through Spontaneous Speech: A Signal Processing Grand
  Challenge*. ICASSP 2023 Signal Processing Grand Challenge.
  [arXiv:2301.05562](https://arxiv.org/abs/2301.05562)
- **DementiaBank Pitt Corpus**: Becker, J. T., Boller, F., Lopez, O. L.,
  Saxton, J., & McGonigle, K. L. (1994). *The Natural History of Alzheimer's
  Disease: Description of Study Cohort and Accuracy of Diagnosis*. Archives
  of Neurology, 51(6), 585-594.
  [Corpus page and required acknowledgements](https://talkbank.org/dementia/access/English/Pitt.html)
  The Pitt corpus requires acknowledgement of NIA grants AG03705 and AG05133.
- **Chou Corpus**: DementiaBank Mandarin Chou corpus, contributed by
  Chia-Ju Chou, containing Mandarin picture-description recordings from
  healthy-control and MCI participants.
  [DementiaBank corpus index](https://talkbank.org/dementia/access/)
- **TAUKADIAL (English and Mandarin/Chinese)**: Luz, S., de la Fuente
  Garcia, S., Haider, F., Fromm, D., MacWhinney, B., Lanzi, A., Chang, Y.-N.,
  Chou, C.-J., & Liu, Y.-C. (2024). *Connected Speech-Based Cognitive
  Assessment in Chinese and English*. Interspeech 2024.
  [doi:10.21437/Interspeech.2024-1807](https://doi.org/10.21437/Interspeech.2024-1807)
- **NCMMSC2021-AD**: *NCMMSC2021 Alzheimer's Disease Recognition Evaluation
  Baseline and Dataset*, Speech and Audio Technology Laboratory, Tsinghua
  University.
  [Dataset page](https://web.ee.tsinghua.edu.cn/satlab/en/gxsj/7552/content/1011.htm)
  and [doi:10.12263/DZXB.20220162](https://doi.org/10.12263/DZXB.20220162)

## Additional Resources

The linguistic feature extractor uses published English and Mandarin
concreteness norms:

- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings
  for 40 thousand generally known English word lemmas. *Behavior Research
  Methods, 46*(3), 904-911.
- Xu, X., & Li, J. (2020). Concreteness/abstractness ratings for two-character
  Chinese words in MELD-SCH. *PLOS ONE, 15*(6), e0232133.
  [doi:10.1371/journal.pone.0232133](https://doi.org/10.1371/journal.pone.0232133)

## Responsible Use

Neurolens AI processes sensitive health-related speech. Users are responsible
for corpus agreements, informed-consent constraints, privacy protection,
secure storage, and applicable institutional or legal review. Predictions are
experimental research outputs, not medical advice.
