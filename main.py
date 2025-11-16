from pathlib import Path
from data_cleanup import cleanup, transcriber

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')

cleanup.normalize(input_file, output_file, verbose=True)
cleanup.denoise(output_file, verbose=True)

print(transcriber.asr(output_file))

