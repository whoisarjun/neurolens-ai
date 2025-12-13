# Main workflow (v2.10)

import json
from pathlib import Path

import torch

from processing import pipeline

# ansi color codes
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
BOLD = '\033[1m'

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

TRAIN_JSON = Path('data_jsons/train.json')
VAL_JSON = Path('data_jsons/val.json')
TEST_JSON = Path('data_jsons/test.json')

# load up jsons
def load_split(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload['data']

# process split
def process_split(json_path: Path, split_name: str, augment=True):
    print(f'\n{BOLD}{CYAN}Loading {split_name} split...{RESET}')

    data = load_split(json_path)

    print(f'{BLUE}   {split_name.title()} samples: {len(data)}{RESET}')
    print(f'\n{BOLD}{CYAN}Cleaning up {split_name} data...{RESET}')

    pipeline.clean_up_all(data)

    print(f'{BOLD}{GREEN}Done cleaning up {split_name} data ✓{RESET}')

    if augment:
        print(f'\n{BOLD}{CYAN}Augmenting {split_name} data...{RESET}')
        data = pipeline.augment_all(data)
        print(f'{BOLD}{GREEN}Done augmenting {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Transcribing {split_name} data...{RESET}')
    pipeline.transcribe_all(data)
    print(f'{BOLD}{GREEN}Done transcribing {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Counting fillers in {split_name} data...{RESET}')
    pipeline.count_fillers_all(data)
    print(f'{BOLD}{GREEN}Done counting fillers in {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Extracting features from {split_name} data...{RESET}')
    pipeline.extract_features_all(data)
    print(f'{BOLD}{GREEN}Done extracting features from {split_name} data ✓{RESET}')

process_split(TRAIN_JSON, 'train', augment=False)
