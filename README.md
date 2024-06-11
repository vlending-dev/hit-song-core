# HIT-SONG-CORE

## 1. Introduction
This is a core library for [hitsong.vlending.kr](https://hitsong.vlending.kr).
It provides functions such as spectrogram generation, model inference, and similar song search.


## 2. Installation
```bash
pip install hit-song-core
```

## 3. Usage

### 1) Generate Spectrogram

```python
from hitsongcore import generate_spectrogram

file_path = 'path/to/file.mp3'
result = generate_spectrogram(file_path, is_force=False)

spectrogram_path = 'path/to/spectrogram/spectrogram.jpg'
slice_path = 'path/to/slice/0.jpg'
```

### 2) Calc Prediction

```python
from hitsongcore import get_model_loader, get_in_memory_storage

file_path = 'path/to/file.mp3'

device = 'cuda'  # 'cpu' or 'cuda'
model_path = 'path/to/model.pt'
num_classes = 8  # example

model_loader = get_model_loader(device, model_path, num_classes)
prediction = model_loader.calc_prediction(file_path)

storage = get_in_memory_storage()
storage.add('1', prediction)

```

### 3) Get Nearest Songs

```python
from hitsongcore import get_in_memory_storage

storage = get_in_memory_storage()

vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

result = storage.query(vector, n=10)
for key, value in result.items():
    print(f'{key}: {value}')
```

## 4. Reference
 - [https://github.com/namngduc/MiRemd](https://github.com/namngduc/MiRemd)

