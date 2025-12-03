import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from processing import cleanup, transcriber
from features import acoustics, linguistics, semantics
from ml import model

TRAIN_JSON = Path('data_jsons/train.json')
TEST_JSON = Path('data_jsons/test.json')

MODEL_WEIGHTS_PATH = Path('models/model_weights.pth')
SCALER_PATH = Path('models/model_scaler.pkl')

# load up jsons
def load_split(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload['data']

# process a SINGLE audio file (GPU-optimized version)
def process_single_item(item):
    try:
        question = item['question']
        input_path = Path(item['input'])
        output_path = Path(item['output'])

        # make sure output dirs exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) cleanup: normalize + denoise
        cleanup.normalize(input_path, output_path, verbose=False)
        cleanup.denoise(output_path, verbose=False)

        # 2) transcribe (GPU-accelerated)
        transcript = transcriber.asr(output_path, verbose=False)

        # 3) extract features
        acoustic_features = acoustics.extract(output_path, transcript, verbose=False)
        linguistic_features = linguistics.extract(transcript, verbose=False)
        semantic_features = semantics.extract(question, transcript, verbose=False)

        # 4) form input vector (shape: (75,))
        input_vector = np.concatenate([
            acoustic_features,
            linguistic_features,
            semantic_features
        ])

        return input_vector, float(item['mmse']), None

    except Exception as e:
        # Return error so we can track failures
        return None, None, str(e)

# Sequential processing (for debugging)
def process_split_sequential(split_data):
    X = []
    y = []

    for item in tqdm(split_data, desc="Processing (sequential)"):
        input_vector, mmse_score, error = process_single_item(item)

        if error is not None:
            print(f"\n⚠️  Error processing {item.get('input')}: {error}")
            continue

        X.append(input_vector)
        y.append(mmse_score)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# load + scale + train + eval
def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Running on CPU.")
        print("Make sure you have:")
        print("  1. NVIDIA drivers installed")
        print("  2. CUDA toolkit installed")
        print("  3. PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # load jsons
    train_data = load_split(TRAIN_JSON)
    test_data = load_split(TEST_JSON)

    # Choose processing mode based on GPU availability
    if torch.cuda.is_available():
        print('\n🚀 GPU detected! Using GPU-optimized processing.')

        # OPTION 1: Threading (recommended for 4090)
        # Allows 4-6 files to process concurrently on GPU
        print('\nProcessing TRAIN split...')
        X_train, y_train = process_split_sequential(train_data)

        print('\nProcessing TEST split...')
        X_test, y_test = process_split_sequential(test_data)
    else:
        print('\n⚠️  No GPU detected. Using sequential processing.')
        X_train, y_train = process_split_sequential(train_data)
        X_test, y_test = process_split_sequential(test_data)

    # scale features
    print('\nFitting scaler on features...')
    model.fit_scaler(X_train)

    X_train_scaled = model.transform_features(X_train)
    X_test_scaled = model.transform_features(X_test)

    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train, batch_size=64)  # Larger batch for GPU
    test_loader = model.create_dataloader(X_test_scaled, y_test, batch_size=64)

    # training time!
    epochs = 15

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print('\n🏋️  Training neural network...')
    for epoch in range(epochs):
        train_loss = model.train_one_epoch(train_loader)
        test_loss, test_accuracy = model.test(test_loader)

        print(
            f'Epoch {epoch + 1:02d} | '
            f'train_loss: {train_loss:.4f} | '
            f'test_loss: {test_loss:.4f} | '
            f'test_acc: {test_accuracy:.4f}'
        )

        model.save(MODEL_WEIGHTS_PATH)
        model.save_scaler(SCALER_PATH)

    print('\n✅ Training complete!')

if __name__ == '__main__':
    main()