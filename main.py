import json
import numpy as np
from pathlib import Path

# from processing import cleanup, transcriber
from features import acoustics, linguistics, llm_scores

# cleanup.normalize(input_file, output_file, verbose=True)
# cleanup.denoise(output_file, verbose=True)

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')
transcript_file = Path('test/transcript.json')

# result = transcriber.asr(output_file, verbose=True)
# with open(transcript_file, 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

with open(transcript_file, 'r', encoding='utf-8') as f:
    transcript = json.load(f)

question = 'So what I\'ll like you to do is describe the Cinderella story'

acoustic_features = acoustics.extract(output_file, transcript, verbose=True)
linguistic_features = linguistics.extract(transcript, verbose=True)
llm_scores = llm_scores.extract(question, transcript, verbose=True)

input_vector: np.array = np.concatenate([
    acoustic_features,
    linguistic_features,
    llm_scores
])
print('Input vector:')
print(input_vector)
print('Shape:')
print(input_vector.shape)

