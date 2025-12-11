# Main workflow

import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from features import acoustics, linguistics, semantics
from ml import model, augmentation
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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

TRAIN_JSON = Path('data_jsons/train.json')
VAL_JSON = Path('data_jsons/val.json')
TEST_JSON = Path('data_jsons/test.json')

MODEL_WEIGHTS_PATH = Path('models/model_weights.pth')
SCALER_PATH = Path('models/model_scaler.pkl')

FEATURE_DIR = Path('models/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

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

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) cleanup
        cleanup.normalize(input_path, output_path, verbose=False)
        cleanup.denoise(output_path, verbose=False)

        # 2) load / augment audio
        if augmentation_mode > 0:
            y, sr = librosa.load(str(output_path), sr=16000, mono=True)

            augmenter = augmentation.DementiaAudioAugmenter(sr=16000)
            y_aug = augmenter.apply_augmentation(y, augmentation_mode)

            TEMP_AUG_DIR.mkdir(parents=True, exist_ok=True)
            temp_aug_path = TEMP_AUG_DIR / f"{output_path.stem}_aug{augmentation_mode}.wav"
            sf.write(temp_aug_path, y_aug, sr)

            processing_path = temp_aug_path
        else:
            processing_path = output_path

        # 3) transcribe (for acoustics + linguistics)
        transcript = transcriber.asr(processing_path, verbose=False)

        # 4) extract acoustic + linguistic + whisper embeddings
        acoustic_features = acoustics.extract(processing_path, transcript, verbose=False)
        linguistic_features = linguistics.extract(transcript, verbose=False)
        whisper_embeddings = transcriber.embeddings(processing_path)[0].cpu().numpy()

        # 5) semantic features with recovery path
        try:
            semantic_features = semantics.extract(question, transcript, output_path, verbose=False)
        except semantics.LLMParseError:
            try:
                # redo ASR from the original cleaned file
                clean_transcript = transcriber.asr(output_path, verbose=False)

                # re-do linguistic features to match the fresh transcript
                linguistic_features = linguistics.extract(clean_transcript, verbose=False)

                # second attempt at extracting semantic features
                semantic_features = semantics.extract(
                    question,
                    clean_transcript,
                    output_path,
                    verbose=False,
                )

            except semantics.LLMParseError:
                print(f"llm parse still failing for {output_path.name}. defaulting to semantic features. 😭")
                semantic_features = semantics.default_semantic_features()

        # 6) form input vector
        input_vector = np.concatenate([
            acoustic_features,
            linguistic_features,
            semantic_features,
            whisper_embeddings
        ])

        if augmentation_mode > 0:
            temp_aug_path.unlink(missing_ok=True)

        return input_vector, float(item['mmse']), None

    except Exception as e:
        return None, None, str(e)

# process training split w augmentation
def process_split_with_augmentation(split_data, augment=True):
    X = []
    y = []

    if augment:
        desc = "Processing with augmentation (4x data)"

        for item in tqdm(split_data, desc=desc):
            # process original + 3 augmented versions
            for aug_mode in [0, 1, 2, 3]:
                input_vector, mmse_score, error = process_single_item(
                    item,
                    augmentation_mode=aug_mode,
                )

                if error is not None:
                    print(f"\n⚠️  error processing {item.get('input')} (aug_mode={aug_mode}): {error}")
                    continue

                X.append(input_vector)
                y.append(mmse_score)
    else:
        desc = "Processing without augmentation"

        for item in tqdm(split_data, desc=desc):
            input_vector, mmse_score, error = process_single_item(
                item,
                augmentation_mode=0,
            )

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

# training with proper train/val/test split
def train(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, epochs=40):
    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train, batch_size=64)
    val_loader = model.create_dataloader(X_val_scaled, y_val, batch_size=64)
    test_loader = model.create_dataloader(X_test_scaled, y_test, batch_size=64)

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{CYAN}Training neural network for {epochs} epochs...{RESET}")
    print(f"{BLUE}Using validation set for early stopping, test set held out until end{RESET}")

    best_val_mae = float('inf')

    for epoch in range(epochs):
        train_loss = model.train_one_epoch(train_loader)
        val_loss, val_mae = model.test(val_loader)

        # reduce learning rate if val mae plateaus
        model.scheduler.step(val_loss)

        # save model if validation MAE improves
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model.save(MODEL_WEIGHTS_PATH)
            model.save_scaler(SCALER_PATH)
            indicator = " 💾"  # indicate save best
        else:
            indicator = ""

        color = GREEN if indicator else BLUE
        print(
            f"{color}epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_mae: {val_mae:.4f}{indicator}{RESET}"
        )

    print(f"\n{GREEN}🎉 Training complete! Best validation MAE: {best_val_mae:.4f}{RESET}")

    # Final evaluation on held-out test set
    print(f"\n{BOLD}{MAGENTA}Evaluating on held-out test set...{RESET}")
    model.load(MODEL_WEIGHTS_PATH)  # Load best model
    test_loss, test_mae = model.test(test_loader)
    print(f"{MAGENTA}Final Test Loss: {test_loss:.4f}{RESET}")
    print(f"{BOLD}{MAGENTA}Final Test MAE: {test_mae:.4f}{RESET}")

# load + scale + train + eval
def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        print(f"{YELLOW}⚠️  CUDA not available – running on CPU{RESET}")
    else:
        print(f"{GREEN}✓ GPU detected – using CUDA acceleration{RESET}")

    # load jsons
    print(f"\n{BOLD}{CYAN}Loading dataset splits...{RESET}")
    train_data = load_split(TRAIN_JSON)
    val_data = load_split(VAL_JSON)
    test_data = load_split(TEST_JSON)
    print(f"{BLUE}   Train samples: {len(train_data)}{RESET}")
    print(f"{BLUE}   Val samples: {len(val_data)}{RESET}")
    print(f"{BLUE}   Test samples: {len(test_data)}{RESET}")

    print(f"\n{BOLD}{CYAN}Processing TRAIN split (augmenting)...{RESET}")
    X_train, y_train = process_split_with_augmentation(train_data, augment=True)
    print(f"   Total train samples after augmentation: {len(X_train)}")

    print(f"\n{BOLD}{CYAN}Checking TRAIN data quality...{RESET}")
    print(f"{BLUE}X_train - NaNs: {np.isnan(X_train).sum()}, Infs: {np.isinf(X_train).sum()}{RESET}")
    print(f"{BLUE}y_train - NaNs: {np.isnan(y_train).sum()}, Infs: {np.isinf(y_train).sum()}{RESET}")
    print(f"{BLUE}X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]{RESET}")
    print(f"{BLUE}y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]{RESET}")

    print(f"\n{BOLD}{CYAN}Processing VAL split (no augments)...{RESET}")
    X_val, y_val = process_split_with_augmentation(val_data, augment=False)
    print(f"   Total val samples: {len(X_val)}")

    print(f"\n{BOLD}{CYAN}Processing TEST split (no augments)...{RESET}")
    X_test, y_test = process_split_with_augmentation(test_data, augment=False)
    print(f"   Total test samples: {len(X_test)}")

    # scale features (fit on train only)
    print(f"\n{BOLD}{CYAN}Fitting scaler on train data...{RESET}")
    model.fit_scaler(X_train)

    X_train_scaled = model.transform_features(X_train)
    X_val_scaled = model.transform_features(X_val)
    X_test_scaled = model.transform_features(X_test)

    # save scaled features + labels
    np.save(FEATURE_DIR / 'X_train_scaled.npy', X_train_scaled)
    np.save(FEATURE_DIR / 'X_val_scaled.npy', X_val_scaled)
    np.save(FEATURE_DIR / 'X_test_scaled.npy', X_test_scaled)
    np.save(FEATURE_DIR / 'y_train.npy', y_train)
    np.save(FEATURE_DIR / 'y_val.npy', y_val)
    np.save(FEATURE_DIR / 'y_test.npy', y_test)

    print(f"\n{GREEN}✓ Saved scaled features + labels to {FEATURE_DIR}{RESET}")

    print(f"{BLUE}X_train_scaled - NaNs: {np.isnan(X_train_scaled).sum()}, Infs: {np.isinf(X_train_scaled).sum()}{RESET}")

    train(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

if __name__ == '__main__':
    main()