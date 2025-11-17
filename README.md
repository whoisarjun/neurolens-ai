# How to use Neurolens AI pipeline:

## 1. cleanup audio
```python
from pathlib import Path
from processing import cleanup

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')

cleanup.normalize(input_file, output_file)
cleanup.denoise(output_file)

# cleaned up audio file saved to test/out.wav
```

## 2. transcribe audio
```python
from processing import transcriber

question = '<insert qn here>'
result = transcriber.asr(output_file)
```
```result``` is a dictionary with the following structure
```json
{
    "text": "<full transcript>",
    "duration": 67,
    "segments": [
        {
            "text": "<segment 1>",
            "start": 0,
            "end": 6.7
        }, ...
    ],
    "filler_count": 67
}
```

## 3. extract features and form input vector
```python
import torch
import numpy as np
from features import acoustics, linguistics, llm_scores

# extract features
acoustic_features = acoustics.extract(output_file, transcript)
linguistic_features = linguistics.extract(transcript)
llm_scores = llm_scores.extract(question, transcript)

# combine into input vector
input_vector = np.concatenate([
    acoustic_features,
    linguistic_features,
    llm_scores
])
```
```input vector``` size: (75,)

## 4. load training and testing batches
```python
from ml import model

# where input vectors are size (75,)
X_train = np.array([input_vector_1, input_vector_2, ...])
y_train = np.array([mmse_score_1, mmse_score_2, ...])
X_test = np.array([input_vector_1, input_vector_2, ...])
y_test = np.array([mmse_score_1, mmse_score_2, ...])

# scale features
model.fit_scaled(X_train)
X_train_scaled = model.transform_features(X_train)
X_test_scaled = model.transform_features(X_test)

# create data loader
train_loader = model.create_dataloader(X_train_scaled, y_train)
test_loader = model.create_dataloader(X_test_scaled, y_test)
```

## 5. train model
```python
epochs = 67

for epoch in range(epochs):
    loss = model.train_one_epoch(train_loader)
    print(f'Epoch {epoch+1} | Loss: {loss:.4f}')
```

## 6. test model and save weights & scaler
```python
loss, accuracy = model.test(test_loader)

model.save('test/model_weights.pth')
model.save_scaler('test/model_scaler.pkl')
```

## 7. load weights & scaler (optional)
```python
model.load('test/model_weights.pth')
model.load_scaler('test/model_scaler.pkl')
```

