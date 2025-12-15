import librosa
import numpy as np

# augmentation to preserve relevant acoustic features but adding realistic variability
class AudioAugmenter:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    # apply diff augmentations (modes 0-3)
    def apply_augmentation(self, y: np.ndarray, augmentation_mode: int) -> np.ndarray:
        # each mode uses different combinations to maximize diversity
        if augmentation_mode == 1:
            y = self._time_stretch(y, rate=np.random.uniform(0.99, 1.01))
            y = self._add_background_noise(y, snr_db=np.random.uniform(30, 40))

        elif augmentation_mode == 2:
            y = self._add_room_reverb(y, room_size='tiny')
            y = self._add_background_noise(y, snr_db=np.random.uniform(28, 38))

        elif augmentation_mode == 3:
            y = self._time_stretch(y, rate=np.random.uniform(0.98, 1.02))

        return y

    # ========== AUGMENTATIONS ========== #

    # time stretch
    def _time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        return librosa.effects.time_stretch(y, rate=rate)

    # realistic room reverberation
    def _add_room_reverb(self, y: np.ndarray, room_size: str = 'small') -> np.ndarray:
        # simple reverb w exponential decay
        if room_size == 'tiny':
            reverb_time = 0.05
            mix = 0.05
        elif room_size == 'small':
            reverb_time = 0.15
            mix = 0.15
        elif room_size == 'medium':
            reverb_time = 0.30
            mix = 0.25
        else:
            reverb_time = 0.10
            mix = 0.10

        # impulse response
        ir_length = int(self.sr * reverb_time)
        ir = np.exp(-5 * np.linspace(0, 1, ir_length))
        ir = ir * np.random.randn(ir_length) * 0.1  # Add randomness

        # convolve with audio
        reverb = np.convolve(y, ir, mode='same')

        # mix
        return (1 - mix) * y + mix * reverb

    # background pink noise
    def _add_background_noise(self, y: np.ndarray, snr_db: float) -> np.ndarray:
        noise = self._generate_pink_noise(len(y))

        # calc signal and noise power
        signal_power = np.mean(y ** 2)
        noise_power = np.mean(noise ** 2)

        # calc required noise power for target snr and scale noise
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)

        return y + noise

    # pink noise generator (helper)
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        # white noise
        white = np.random.randn(length)

        # pink noise filter (approx w fft)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)

        # apply 1/f scaling
        scaling = 1 / np.sqrt(freqs + 1e-10)
        fft_pink = fft * scaling

        pink = np.fft.irfft(fft_pink, n=length)

        # norm
        pink = pink / np.max(np.abs(pink))

        return pink

# wraps dataloader to apply augs on the fly
class AugmentedDataLoader:
    def __init__(self, original_data, augment_multiplier: int = 3):
        self.original_data = original_data
        self.augment_multiplier = augment_multiplier
        self.augmenter = AudioAugmenter(sr=16000)

    def __len__(self):
        return len(self.original_data) * (1 + self.augment_multiplier)

    def __getitem__(self, idx):
        original_idx = idx % len(self.original_data)
        aug_mode = idx // len(self.original_data)  # 0=original, 1-3=augmented

        item = self.original_data[original_idx].copy()
        item['augmentation_mode'] = aug_mode

        return item

    def get_all_items(self):
        all_items = []
        for idx in range(len(self)):
            all_items.append(self.__getitem__(idx))
        return all_items

# ===== INTEGRATION HELPERS ===== #

def augment_audio_file(input_path: str, augmentation_mode: int, sr: int = 16000) -> np.ndarray:
    augmenter = AudioAugmenter(sr=sr)

    # load audio
    y, _ = librosa.load(input_path, sr=sr, mono=True)

    # apply augmentation
    y_aug = augmenter.apply_augmentation(y, augmentation_mode)

    return y_aug

def create_augmented_split(original_data_path: str, multiplier: int = 3):
    import json
    from pathlib import Path

    with Path(original_data_path).open('r') as f:
        data = json.load(f)

    original_items = data['data']

    return AugmentedDataLoader(original_items, augment_multiplier=multiplier)