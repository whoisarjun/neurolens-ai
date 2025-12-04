import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import librosa
import soundfile as sf
import shutil

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from ml import model, augmentation
from processing import cleanup, transcriber
from features import acoustics, linguistics, semantics

TRAIN_JSON = Path('data_jsons/train.json')
TEST_JSON = Path('data_jsons/test.json')

MODEL_WEIGHTS_PATH = Path('models/model_weights.pth')
SCALER_PATH = Path('models/model_scaler.pkl')

TEMP_AUG_DIR = Path('temp_augmented')

# load up jsons
def load_split(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload['data']

# process a single audio file (w augmentation)
def process_single_item(item, augmentation_mode=0):
    try:
        question = item['question']
        input_path = Path(item['input'])
        output_path = Path(item['output'])

        # make sure output dirs exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) cleanup: normalize + denoise
        cleanup.normalize(input_path, output_path, verbose=False)
        cleanup.denoise(output_path, verbose=False)

        if augmentation_mode > 0:
            # Load cleaned audio
            y, sr = librosa.load(str(output_path), sr=16000, mono=True)

            # Apply augmentation
            augmenter = augmentation.DementiaAudioAugmenter(sr=16000)
            y_aug = augmenter.apply_augmentation(y, augmentation_mode)

            # Save to temp location
            TEMP_AUG_DIR.mkdir(parents=True, exist_ok=True)
            temp_aug_path = TEMP_AUG_DIR / f"{output_path.stem}_aug{augmentation_mode}.wav"
            sf.write(temp_aug_path, y_aug, sr)

            # Use augmented file for feature extraction
            processing_path = temp_aug_path
        else:
            # Use original cleaned file
            processing_path = output_path
        # ================================================

        # 2) transcribe (GPU-accelerated)
        transcript = transcriber.asr(processing_path, verbose=False)

        # 3) extract features
        acoustic_features = acoustics.extract(processing_path, transcript, verbose=False)
        linguistic_features = linguistics.extract(transcript, verbose=False)
        semantic_features = semantics.extract(question, transcript, verbose=False)

        # 4) form input vector
        input_vector = np.concatenate([
            acoustic_features,
            linguistic_features,
            semantic_features
        ])

        # ============ NEW: Cleanup temp file ============
        if augmentation_mode > 0:
            temp_aug_path.unlink(missing_ok=True)
        # ===============================================

        return input_vector, float(item['mmse']), None

    except Exception as e:
        return None, None, str(e)

# process training split w augmentation
def process_split_with_augmentation(split_data, augment=True):
    X = []
    y = []

    if augment:
        # Create 4x data: original + 3 augmented versions
        desc = "Processing with augmentation (4x data)"

        for item in tqdm(split_data, desc=desc):
            # Process original (mode 0)
            for aug_mode in [0, 1, 2, 3]:
                input_vector, mmse_score, error = process_single_item(item, augmentation_mode=aug_mode)

                if error is not None:
                    print(f"\n⚠️  error processing {item.get('input')} (aug_mode={aug_mode}): {error}")
                    continue

                X.append(input_vector)
                y.append(mmse_score)
    else:
        desc = "Processing without augmentation"

        for item in tqdm(split_data, desc=desc):
            input_vector, mmse_score, error = process_single_item(item, augmentation_mode=0)

            if error is not None:
                print(f"\n⚠️  Error processing {item.get('input')}: {error}")
                continue

            X.append(input_vector)
            y.append(mmse_score)

    if TEMP_AUG_DIR.exists():
        shutil.rmtree(TEMP_AUG_DIR, ignore_errors=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

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
    else:
        print("🚀 GPU detected! Using CUDA acceleration.")

    # load jsons
    print("\n📁 Loading dataset splits...")
    train_data = load_split(TRAIN_JSON)
    test_data = load_split(TEST_JSON)
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")

    print('\n🔊 Processing TRAIN split (WITH augmentation)...')
    X_train, y_train = process_split_with_augmentation(train_data, augment=True)
    print(f"   Total train samples after augmentation: {len(X_train)}")

    print('\n🔊 Processing TEST split (NO augmentation)...')
    X_test, y_test = process_split_with_augmentation(test_data, augment=False)
    print(f"   Total test samples: {len(X_test)}")

    # scale features
    print('\n📊 Fitting scaler on training features...')
    model.fit_scaler(X_train)

    X_train_scaled = model.transform_features(X_train)
    X_test_scaled = model.transform_features(X_test)

    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train, batch_size=64)
    test_loader = model.create_dataloader(X_test_scaled, y_test, batch_size=64)

    # training time!
    epochs = 15

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f'\n🏋️  Training neural network for {epochs} epochs...')

    best_mae = float('inf')

    for epoch in range(epochs):
        train_loss = model.train_one_epoch(train_loader)
        test_loss, test_accuracy = model.test(test_loader)

        if test_accuracy < best_mae:
            best_mae = test_accuracy
            model.save(MODEL_WEIGHTS_PATH)
            model.save_scaler(SCALER_PATH)
            indicator = " 💾 [saved best]"
        else:
            indicator = ""

        print(
            f'Epoch {epoch + 1:02d}/{epochs} | '
            f'train_loss: {train_loss:.4f} | '
            f'test_loss: {test_loss:.4f} | '
            f'test_MAE: {test_accuracy:.4f}{indicator}'
        )

    print(f'\n✅ Training complete! Best test MAE: {best_mae:.4f}')

if __name__ == '__main__':
    main()