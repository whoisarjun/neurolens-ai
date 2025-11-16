from pathlib import Path
import json
# from processing import cleanup, transcriber
from features import acoustics

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')

with open(Path('test/transcript.json'), 'r', encoding='utf-8') as f:
    transcript = json.load(f)
# cleanup.normalize(input_file, output_file, verbose=True)
# cleanup.denoise(output_file, verbose=True)

# result = transcriber.asr(output_file, verbose=True)
# with open(transcript, 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

print(acoustics.extract(output_file, transcript))
