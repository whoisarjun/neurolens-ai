# Main workflow (v3.0)

import json
from pathlib import Path

import numpy as np
import torch

from ml import model
from processing import pipeline
from utils import cache

# ansi color codes
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
BOLD = '\033[1m'

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

TRAIN_JSON = Path('data_jsons/train.json')
VAL_JSON = Path('data_jsons/val.json')
TEST_JSON = Path('data_jsons/test.json')

REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
EMB_SCALER_PATH = Path('models/model_emb_scaler.pkl')
EMB_PCA_PATH = Path('models/model_emb_pca.pkl')

FEATURE_DIR = Path('models/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MISSING_MMSE = np.nan
MISSING_DIAGNOSIS = -1

# load up jsons
def load_split(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload['data']

def parse_mmse(value, sample_name: str):
    if value is None:
        return MISSING_MMSE, False

    try:
        return float(value), True
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid mmse for sample "{sample_name}": {value!r}') from exc

def parse_diagnosis(value, sample_name: str):
    if value is None:
        return MISSING_DIAGNOSIS, False

    if value not in model.cog_statuses:
        raise ValueError(
            f'Invalid diagnosis for sample "{sample_name}": {value!r}. '
            f'Expected one of {model.cog_statuses}.'
        )

    return model.cog_statuses.index(value), True

# process split
def process_split(json_path: Path, split_name: str, use_cache=None, augment=True):
    if use_cache is None:
        use_cache = {
            'transcript': True,
            'acoustics': False,
            'linguistics': False,
            'semantics': True,
            'embeddings': True
        }

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
    pipeline.transcribe_all(data, use_cache_transcripts=use_cache['transcript'])
    print(f'{BOLD}{GREEN}Done transcribing {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Extracting features from {split_name} data...{RESET}')
    pipeline.extract_features_all(data, use_cache_acoustics=use_cache['acoustics'], use_cache_linguistics=use_cache['linguistics'], use_cache_semantics=use_cache['semantics'])
    print(f'{BOLD}{GREEN}Done extracting features from {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Generating embeddings from {split_name} data...{RESET}')
    pipeline.gen_embeddings_all(data, use_cache_embeddings=use_cache['embeddings'], batch_size=4)
    print(f'{BOLD}{GREEN}Done generating embeddings from {split_name} data ✓{RESET}')

    X_features = [d['features'] for d in data]
    X_embeddings = [d['embeddings'] for d in data]
    y = []
    z = []
    y_mask = []
    z_mask = []

    for data_item in data:
        sample_name = str(data_item.get('output', data_item.get('input', 'unknown sample')))
        mmse_value, has_mmse = parse_mmse(data_item.get('mmse'), sample_name)
        diagnosis_value, has_diagnosis = parse_diagnosis(data_item.get('diagnosis'), sample_name)

        y.append(mmse_value)
        z.append(diagnosis_value)
        y_mask.append(has_mmse)
        z_mask.append(has_diagnosis)

    return X_features, X_embeddings, y, z, y_mask, z_mask

# training with proper train/val/test split
def train(
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test,
    z_train, z_val, z_test,
    y_train_mask, y_val_mask, y_test_mask,
    z_train_mask, z_val_mask, z_test_mask,
    regressor, classifier,
    reg_criterion, cls_criterion,
    optimizer, scheduler,
    lam=0.5,
    epochs=50,
    verbose=True
):
    train_loader = model.create_dataloader(
        X_train_scaled, y_train, z_train, y_train_mask, z_train_mask, batch_size=64
    )
    val_loader = model.create_dataloader(
        X_val_scaled, y_val, z_val, y_val_mask, z_val_mask, batch_size=64
    )
    test_loader = model.create_dataloader(
        X_test_scaled, y_test, z_test, y_test_mask, z_test_mask, batch_size=64
    )

    best_score = -float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        train_loss, train_mmse, train_cog = model.train_mt_one_epoch(
            train_loader,
            regressor, classifier,
            reg_criterion, cls_criterion,
            optimizer,
            lam=lam
        )

        val_reg_loss, val_mae, val_rmse = model.test_reg(val_loader, regressor, reg_criterion)
        val_cls_loss, val_acc, val_f1, _ = model.test_cls(val_loader, classifier, cls_criterion)

        scheduler_target = val_reg_loss if val_reg_loss is not None else train_mmse
        if scheduler_target is not None:
            scheduler.step(scheduler_target)

        score_parts = []
        if val_mae is not None:
            score_parts.append(-val_mae)
        if val_f1 is not None:
            score_parts.append(2 * val_f1)
        if not score_parts:
            raise RuntimeError('Validation split has no MMSE or diagnosis labels to score.')
        score = sum(score_parts)

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1

            # Save both heads (they share backbone weights anyway)
            model.save(REG_WEIGHTS_PATH, regressor)
            model.save(CLS_WEIGHTS_PATH, classifier)
            indicator = " 💾"
        else:
            indicator = ""

        if verbose:
            color = GREEN if indicator else BLUE
            print(
                f"{color}mt epoch {epoch+1:02d}/{epochs} | "
                f"train_total: {model.format_metric(train_loss)} "
                f"(mmse {model.format_metric(train_mmse)}, cog {model.format_metric(train_cog)}) | "
                f"val_mae: {model.format_metric(val_mae)} | val_rmse: {model.format_metric(val_rmse)} | "
                f"val_f1: {model.format_metric(val_f1)} | val_acc: {model.format_metric(val_acc)}{indicator}{RESET}"
            )

    if verbose:
        print(f"\n{GREEN}🎉 Multitask training done. Best score at epoch {best_epoch}{RESET}")

    # final test eval
    model.load(REG_WEIGHTS_PATH, regressor)
    model.load(CLS_WEIGHTS_PATH, classifier)

    test_reg_loss, test_mae, test_rmse = model.test_reg(test_loader, regressor, reg_criterion)
    test_cls_loss, test_acc, test_f1, test_cm = model.test_cls(test_loader, classifier, cls_criterion)

    return test_mae, test_rmse, test_acc, test_f1, test_cm

# main flow
def main():
    use_cache_list = {
        'transcript': cache.ask('transcripts'),
        'acoustics': cache.ask('acoustic features'),
        'linguistics': cache.ask('linguistic features'),
        'semantics': cache.ask('LLM-generated semantic features'),
        'embeddings': cache.ask('audio embeddings')
    }

    X_train, X_train_emb, y_train, z_train, y_train_mask, z_train_mask = process_split(TRAIN_JSON, 'train', use_cache=use_cache_list, augment=True)
    X_val, X_val_emb, y_val, z_val, y_val_mask, z_val_mask = process_split(VAL_JSON, 'validation', use_cache=use_cache_list, augment=False)
    X_test, X_test_emb, y_test, z_test, y_test_mask, z_test_mask = process_split(TEST_JSON, 'test', use_cache=use_cache_list, augment=False)
    print(f"\n{GREEN}Done processing all data!{RESET}")

    # scale features (fit on train only)
    print(f"\n{BOLD}{CYAN}Fit-transforming scalers and PCA on train data...{RESET}")
    model.fit_scaler(X_train)
    model.fit_emb_scaler(X_train_emb)
    model.save_scaler(SCALER_PATH)
    model.save_emb_scaler(EMB_SCALER_PATH)

    X_train_emb = model.transform_emb_scaler(X_train_emb)
    X_val_emb = model.transform_emb_scaler(X_val_emb)
    X_test_emb = model.transform_emb_scaler(X_test_emb)

    model.fit_emb_pca(X_train_emb)
    model.save_emb_pca(EMB_PCA_PATH)

    X_train_scaled = np.concatenate([model.transform_features(X_train), model.transform_emb_pca(X_train_emb)], axis=1)
    X_val_scaled = np.concatenate([model.transform_features(X_val), model.transform_emb_pca(X_val_emb)], axis=1)
    X_test_scaled = np.concatenate([model.transform_features(X_test), model.transform_emb_pca(X_test_emb)], axis=1)

    # save scaled features + labels
    np.save(FEATURE_DIR / 'X_train_scaled.npy', X_train_scaled)
    np.save(FEATURE_DIR / 'X_val_scaled.npy', X_val_scaled)
    np.save(FEATURE_DIR / 'X_test_scaled.npy', X_test_scaled)
    np.save(FEATURE_DIR / 'y_train.npy', y_train)
    np.save(FEATURE_DIR / 'y_val.npy', y_val)
    np.save(FEATURE_DIR / 'y_test.npy', y_test)
    np.save(FEATURE_DIR / 'z_train.npy', z_train)
    np.save(FEATURE_DIR / 'z_val.npy', z_val)
    np.save(FEATURE_DIR / 'z_test.npy', z_test)
    np.save(FEATURE_DIR / 'y_train_mask.npy', y_train_mask)
    np.save(FEATURE_DIR / 'y_val_mask.npy', y_val_mask)
    np.save(FEATURE_DIR / 'y_test_mask.npy', y_test_mask)
    np.save(FEATURE_DIR / 'z_train_mask.npy', z_train_mask)
    np.save(FEATURE_DIR / 'z_val_mask.npy', z_val_mask)
    np.save(FEATURE_DIR / 'z_test_mask.npy', z_test_mask)

    print(f"\n{GREEN}✓ Saved scaled features + labels to {FEATURE_DIR}{RESET}")

    backbone = model.new_backbone()
    regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler = model.new_multitask(backbone)

    test_mae, test_rmse, test_acc, test_f1, test_cm = train(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        z_train, z_val, z_test,
        y_train_mask, y_val_mask, y_test_mask,
        z_train_mask, z_val_mask, z_test_mask,
        regressor, classifier,
        reg_criterion, cls_criterion,
        optimizer, scheduler,
        lam=0.45,
        epochs=50
    )

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}FINAL TEST SET RESULTS{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{YELLOW}Regression → MAE: {model.format_metric(test_mae)} | RMSE: {model.format_metric(test_rmse)}{RESET}")
    print(f"{YELLOW}Classification → Accuracy: {model.format_metric(test_acc)} | F1: {model.format_metric(test_f1)}{RESET}")
    print(f"{CYAN}Confusion Matrix:\n{test_cm if test_cm is not None else 'n/a'}{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}\n")

if __name__ == '__main__':
    main()
