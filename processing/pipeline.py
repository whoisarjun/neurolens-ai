# Centralizaed script for stage-wise batch processing (memory efficiency)

import os
import re
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from features import acoustics, linguistics, semantics
from ml import augmentation
from processing import cleanup, transcriber

# ansi color codes
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"

# ========== STAGE 1 ========== #
def clean_up_all(all_data: list):
    desc = 'Cleaning up audio'

    for data in tqdm(all_data, desc=desc):
        from_fp = Path(data['input'])
        to_fp = Path(data['output'])

        to_fp.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(to_fp):
            continue

        cleanup.normalize(from_fp, to_fp, verbose=False)
        cleanup.denoise(to_fp, verbose=False)

# ========== STAGE 2 ========== #
def augment_all(all_data: list):
    desc = 'Augmenting audio files'
    augmented_list = all_data.copy()

    for data in tqdm(all_data, desc=desc):
        fp = Path(data['output'])

        if not os.path.exists(fp):
            raise FileNotFoundError(f'Cannot augment nonexistent file: {str(fp)}')

        y, sr = librosa.load(str(fp), sr=16000, mono=True)
        augmenter = augmentation.AudioAugmenter(sr=16000)

        for aug_mode in range(1, 4):
            y_aug = augmenter.apply_augmentation(y, aug_mode)

            aug_fp = fp.with_name(f'{fp.stem}_aug{aug_mode}{fp.suffix}')
            sf.write(aug_fp, y_aug, sr)

            aug_data = deepcopy(data)
            aug_data['output'] = str(aug_fp)

            augmented_list.append(aug_data)

    return augmented_list

# ========== STAGE 3 ========== #
def transcribe_all(all_data: list):
    desc = 'Transcribing audio'

    transcriber.get_whisper()
    transcript_cache = {}  # base_fp -> transcript

    for data in tqdm(all_data, desc=desc):
        fp = Path(data['output'])

        if not fp.exists():
            raise FileNotFoundError(f'Cannot transcribe nonexistent audio: {fp}')

        # detect augmentation
        m = re.match(r'(.+)_aug\d+$', fp.stem)

        if m:
            # this is an augmented file → reuse base transcript
            base_fp = fp.with_name(m.group(1) + fp.suffix)

            if base_fp not in transcript_cache:
                raise RuntimeError(
                    f'Base transcript missing for {fp.name}. '
                    f'Make sure base files come before augmentations.'
                )

            data['transcript'] = transcript_cache[base_fp]

        else:
            # base file → do ASR once
            transcript = transcriber.asr(fp)
            data['transcript'] = transcript
            transcript_cache[fp] = transcript

    transcriber.unload_models()

# ========== STAGE 4 ========== #
def count_fillers_all(all_data: list):
    desc = 'Counting fillers'

    transcriber.get_crisper()
    filler_cache = {}  # base_fp -> filler_count

    for data in tqdm(all_data, desc=desc):
        fp = Path(data['output'])

        if not fp.exists():
            raise FileNotFoundError(f'Cannot transcribe nonexistent audio: {fp}')

        m = re.match(r'(.+)_aug\d+$', fp.stem)

        if m:
            # augmented → reuse base
            base_fp = fp.with_name(m.group(1) + fp.suffix)

            if base_fp not in filler_cache:
                raise RuntimeError(
                    f'Base filler count missing for {fp.name}. '
                    f'Make sure base files come before augmentations.'
                )

            data['transcript']['filler_count'] = filler_cache[base_fp]

        else:
            # base → compute once
            filler_count = transcriber.filler_count(fp)
            data['transcript']['filler_count'] = filler_count
            filler_cache[fp] = filler_count

    transcriber.unload_models()

# ========== STAGE 5 ========== #
def extract_features_all(all_data: list):
    desc = 'Extracting features'

    for data in tqdm(all_data, desc=desc):
        fp = Path(data['output'])
        question = data['question']
        transcript = data['transcript']

        if not os.path.exists(fp):
            raise FileNotFoundError(f'Cannot extract features from nonexistent audio: {str(fp)}')

        acoustic_features = acoustics.extract(fp, transcript, verbose=False)
        linguistic_features = linguistics.extract(transcript, verbose=False)

        base_fp = fp.with_name(
            re.sub(r'_aug\d+$', '', fp.stem) + fp.suffix
        )
        try:
            semantic_features = semantics.extract(question, transcript, base_fp, verbose=False)
        except semantics.LLMParseError:
            try:
                # redo ASR and linguistic features from the original cleaned file for 2nd semantic features attempt
                transcriber.get_whisper()
                clean_transcript = transcriber.asr(base_fp, verbose=False)
                transcriber.unload_models()

                linguistic_features = linguistics.extract(clean_transcript, verbose=False)
                semantic_features = semantics.extract(question, transcript, base_fp, verbose=False)
            except semantics.LLMParseError:
                print(f"LLM parse still failing for {base_fp.name}. setting default semantic features. 😭")
                semantic_features = semantics.default_semantic_features()

        features = np.concatenate([
            acoustic_features,
            linguistic_features,
            semantic_features
        ]).astype(np.float32)
        data['features'] = features

# ========== STAGE 6 ========== #
def gen_embeddings_all(all_data: list):
    desc = 'Generating audio embeddings'

    transcriber.get_hubert()
    for data in tqdm(all_data, desc=desc):
        fp = data['output']

        if not os.path.exists(fp):
            raise FileNotFoundError(f'Cannot generate embeddings from nonexistent audio: {str(fp)}')

        embeddings = transcriber.embeddings(fp)
        embeddings_np = embeddings.numpy().ravel().astype(np.float32)
        data['features'] = np.concatenate([
            data['features'],
            embeddings_np
        ])

    transcriber.unload_models()
