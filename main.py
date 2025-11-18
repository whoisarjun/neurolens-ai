import json
from pathlib import Path
from tqdm import tqdm

import numpy as np

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

# process data (cleaup + transcript + extract features)
def process_split(split_data):
    X = []
    y = []

    for item in tqdm(split_data):
        question = item['question']
        input_path = Path(item['input'])
        output_path = Path(item['output'])

        # make sure output dirs exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) cleanup: normalize + denoise
        cleanup.normalize(input_path, output_path)
        cleanup.denoise(output_path)

        # 2) transcribe
        transcript = transcriber.asr(output_path)
        # transcript is the dict with text, segments, etc.

        # 3) extract features
        acoustic_features = acoustics.extract(output_path, transcript)
        linguistic_features = linguistics.extract(transcript)
        llm_feature_vec = llm_scores.extract(question, transcript)

        # 4) form input vector (shape: (75,))
        input_vector = np.concatenate([
            acoustic_features,
            linguistic_features,
            llm_feature_vec
        ])

        X.append(input_vector)
        y.append(float(item['mmse']))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# load + scale + train + eval
def main():
    # load jsons
    train_data = load_split(TRAIN_JSON)
    test_data = load_split(TEST_JSON)

    # process data (cleanup + transcript + extract features)
    print('Processing TRAIN split...')
    X_train, y_train = process_split(train_data)

    print('Processing TEST split...')
    X_test, y_test = process_split(test_data)

    # sacle features
    print('Fitting scaler on features...')
    model.fit_scaler(X_train)

    X_train_scaled = model.transform_features(X_train)
    X_test_scaled = model.transform_features(X_test)

    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train)
    test_loader = model.create_dataloader(X_test_scaled, y_test)

    # training time!
    epochs = 15

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

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

        print(f'Saved!')

if __name__ == '__main__':
    main()