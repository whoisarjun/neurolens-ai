# How to use Neurolens pipeline:

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
import numpy as np
from features import acoustics, linguistics, llm_scores

acoustic_features = acoustics.extract(output_file, transcript)
linguistic_features = linguistics.extract(transcript)
llm_scores = llm_scores.extract(question, transcript)

input_vector: np.array = np.concatenate([
    acoustic_features,
    linguistic_features,
    llm_scores
])
```

Input vector size: (75,)

