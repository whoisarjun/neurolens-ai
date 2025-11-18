import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from processing import cleanup, transcriber
from features import acoustics, linguistics, llm_scores
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
        llm_feature_vec = llm_scores.extract(question, transcript, verbose=False)

        # 4) form input vector (shape: (75,))
        input_vector = np.concatenate([
            acoustic_features,
            linguistic_features,
            llm_feature_vec
        ])

        return input_vector, float(item['mmse']), None

    except Exception as e:
        # Return error so we can track failures
        return None, None, str(e)


# GPU-optimized processing with threading
def process_split_gpu(split_data, max_workers=5):
    print(f'Using {max_workers} parallel threads (GPU mode)')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available!"}')

    X = []
    y = []
    errors = []

    # Use ThreadPoolExecutor (threads share GPU memory)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_item, item): item
            for item in split_data
        }

        # Collect results as they complete (with progress bar)
        for future in tqdm(as_completed(future_to_item), total=len(split_data), desc="Processing (GPU)"):
            item = future_to_item[future]
            try:
                input_vector, mmse_score, error = future.result()

                if error is not None:
                    errors.append({
                        'file': item.get('input'),
                        'error': error
                    })
                    print(f"\n⚠️  Error processing {item.get('input')}: {error}")
                else:
                    X.append(input_vector)
                    y.append(mmse_score)

            except Exception as e:
                errors.append({
                    'file': item.get('input'),
                    'error': str(e)
                })
                print(f"\n⚠️  Unexpected error processing {item.get('input')}: {e}")

    # Report any errors
    if errors:
        print(f"\n⚠️  {len(errors)} files failed to process:")
        for err in errors[:5]:  # Show first 5
            print(f"  - {err['file']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# Sequential processing (for debugging)
def process_split_sequential(split_data):
    """
    Original sequential processing (for debugging or if parallel fails).
    """
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
        X_train, y_train = process_split_gpu(train_data, max_workers=6)

        print('\nProcessing TEST split...')
        X_test, y_test = process_split_gpu(test_data, max_workers=6)
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