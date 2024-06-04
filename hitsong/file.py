from pathlib import Path

import librosa
from librosa.feature import melspectrogram

import matplotlib.pyplot as plt
from PIL import Image


class File:
    def __init__(self, file_path, index=None):
        self._index = index
        self.path = Path(file_path)

    def index(self):
        return self._index or str(self.path)


class Audio(File):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectrogram_path = self.path.parent / 'spectrogram'
        self.spectrogram_file_name = 'spectrogram.jpg'
        self.slice_path = self.path.parent / 'slice'

    def index(self):
        return self._index or str(self.path)

    def exists_audio_file(self):
        return self.path.exists()

    def exists_spectrogram(self):
        return (self.spectrogram_path / self.spectrogram_file_name).exists()

    def exists_slice(self):
        return self.slice_path.exists()

    def slice_files(self):
        return [file for file in self.slice_path.iterdir() if file.is_file() and file.suffix == '.jpg']

    def mkdir(self):
        self.spectrogram_path.mkdir(parents=True, exist_ok=True)
        self.slice_path.mkdir(parents=True, exist_ok=True)

    def clear_spectrogram(self):
        spectrogram_file = self.spectrogram_path / self.spectrogram_file_name
        spectrogram_file.unlink(missing_ok=True)

    def clear_slice(self):
        for f in self.slice_path.iterdir():
            f.unlink(missing_ok=True)

    def clear_all(self):
        self.clear_spectrogram()
        self.clear_slice()

    def generate_spectrogram_slice(self):
        y, sr = librosa.load(self.path)
        mel_spectrogram_array = melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel = librosa.power_to_db(mel_spectrogram_array)

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = float(mel.shape[1]) / float(100)
        fig_size[1] = float(mel.shape[0]) / float(100)
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis('off')
        plt.axes((0., 0., 1., 1.0), frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(mel, cmap='gray_r')

        plt.savefig(f'{self.spectrogram_path}/{self.spectrogram_file_name}', dpi=100)
        plt.close()

        image = Image.open(f'{self.spectrogram_path}/{self.spectrogram_file_name}')
        subsample_size = 128

        # Take 20 frames from the 6th frame to calculate the cosine similarity
        number_of_samples = 26

        counter = 0
        for i in range(6, number_of_samples):
            start = i * subsample_size
            image_temp = image.crop((start, 0, start + subsample_size, subsample_size))
            slice_file_name = f'{self.slice_path}/{counter}.jpg'
            image_temp.save(slice_file_name)
            counter += 1
