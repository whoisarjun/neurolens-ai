# Cache managers for time-consuming parts of the pipeline

import hashlib
import pickle
from pathlib import Path

def key(filename: Path, cache_dir: Path, variant: str | None = None):
    head = str(filename).split('DATA/')[-1]
    payload = f'{variant}::{head}' if variant else head
    cache_key = hashlib.md5(payload.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.pkl"
    return cache_file

def load(cache_file: Path):
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save(cache_file: Path, item):
    with open(cache_file, 'wb') as f:
        pickle.dump(item, f)

def ask(folder: str) -> bool:
    resp = input(f'Use cache for {folder}? (y/n): ') == 'y'
    print(f'{"U" if resp else "Not u"}sing cache for {folder}\n')
    return resp
