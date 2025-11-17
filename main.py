import json
import torch
import numpy as np
from pathlib import Path
from ml import model
from features import acoustics, linguistics, llm_scores

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')
transcript_file = Path('test/transcript.json')

with open(transcript_file, 'r', encoding='utf-8') as f:
    transcript = json.load(f)

question = 'So what I\'ll like you to do is describe the Cinderella story'

acoustic_features = acoustics.extract(output_file, transcript, verbose=True)
linguistic_features = linguistics.extract(transcript, verbose=True)
llm_scores = llm_scores.extract(question, transcript, verbose=True)

input_vector = np.concatenate([
    acoustic_features,
    linguistic_features,
    llm_scores
])

# sample training x and y
X_train = np.array([input_vector])  # Shape: [N, 75]
y_train = np.array([24.0])  # Shape: [N]

# scale x
model.fit_scaler(X_train)
X_train_scaled = model.transform_features(X_train)

# load data and train model
train_loader = model.create_dataloader(X_train_scaled, y_train, batch_size=4)

for epoch in range(10):
    loss = model.train_one_epoch(train_loader)
    print(f'Epoch {epoch+1}/10 | Loss: {loss:.4f}')

# save everything
model.save(Path('test/model_weights.pth'))
model.save_scaler(Path('test/scaler.pkl'))

