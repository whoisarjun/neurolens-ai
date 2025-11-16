from pathlib import Path
from data_cleanup import cleanup

input_file = Path('test/conversation.mp3')
output_file = Path('test/out.wav')

cleanup.normalize(input_file, output_file)
cleanup.denoise(output_file)
