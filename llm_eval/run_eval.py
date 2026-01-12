# Evaluation of LLMs for extraction of semantic features

import json
import os
import re
from pathlib import Path

from tqdm import tqdm

from features import semantics
from main import BOLD, CYAN, GREEN, RESET
from processing import pipeline, transcriber

models = [
    'qwen3:8b',
    'deepseek-r1:8b',
    'llama3.1:8b',
    'granite3.3:8b',
    'ministral-3:8b'
]

AUDIO_EXTS = ['.mp3', '.wav']
DATA_DIR = Path('EVAL_DATA')

def save_result(model, fn, result):
    file = Path('models/features') / f'{re.match(r"[A-Za-z]+", model).group(0)}.json'
    data = []
    if file.exists():
        with file.open() as f:
            data = json.load(f)['results']
    else:
        with file.open('w') as f:
            json.dump({'results': []}, f, indent=4)
            data = []
    if len(data) == 0 or not any(d['file'] == fn for d in data):
        data.append({
            'file': fn,
            'results': []
        })
    entry = next(d for d in data if d['file'] == fn)
    entry['results'].append(result)
    with file.open('w') as f:
        json.dump({'results': data}, f, indent=4)

def exists(fn, model, attempt):
    file = Path('models/features') / f'{re.match(r"[A-Za-z]+", model).group(0)}.json'
    if file.exists():
        with file.open() as f:
            data = json.load(f)['results']
    else:
        return False
    if len(data) == 0 or not any(d['file'] == fn for d in data):
        return False
    entry = next(d for d in data if d['file'] == fn)
    lst = entry['results']
    return attempt < len(lst)

def extract(question, transcript, base_fp, model):
    try:
        semantic_features = semantics.extract(question, transcript, base_fp, use_cache=False, model=model)
    except semantics.LLMParseError:
        print('PARSE ERROR')
        try:
            semantic_features = semantics.extract(question, transcript, base_fp, use_cache=False, model=model)
        except semantics.LLMParseError:
            semantic_features = semantics.default_semantic_features()
    return semantic_features.tolist()

def main():
    data = [{'output': str(f), 'question': 'Tell me everything you see going on in the picture in front of you.'} for f in sorted([
        p for p in DATA_DIR.rglob('*')
        if p.suffix.lower() in AUDIO_EXTS and p.is_file()
    ])]
    print(f'\n{BOLD}{CYAN}Transcribing evaluation data...{RESET}')
    pipeline.transcrlmiibe_all(data, use_cache_transcripts=True, recycle_augs=False)
    print(f'{BOLD}{GREEN}Done transcribing evaluation data ✓{RESET}')

    for d in tqdm(data, desc='Progress: ', position=0):
        qn = d['question']
        file = d['output']
        transcript = d['transcript']
        for m in tqdm(models, desc='Models: ', position=1, leave=False):
            for i in tqdm(range(5), desc='Attempt: ', position=2, leave=False):
                if not exists(file, m, i):
                    features = extract(qn, transcript, file, m)
                    save_result(m, file, features)

main()



