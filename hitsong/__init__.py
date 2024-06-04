__version__ = '0.0.1'


from .file import Audio
from .model import ModelLoader
from .storages import BaseStorage, InMemoryStorage, ChromaDBStorage, ChromaDBMode


def generate_spectrogram(audio_file_path: str, is_force: bool = False):
    audio = Audio(audio_file_path)

    if not audio.exists_audio_file():
        return False

    if not audio.exists_spectrogram() or is_force:
        audio.mkdir()
        audio.clear_all()
        audio.generate_spectrogram_slice()
    return True


def calc_prediction(loader: ModelLoader, audio_files: dict, storage: BaseStorage):
    save_data = dict()
    for _i, audio_file_path in audio_files.items():
        audio = Audio(audio_file_path)
        prediction = loader.calc_prediction(audio)
        if prediction is not None:
            save_data[_i] = prediction
    storage.multi_add(save_data)


def get_nearest(vector, storage: BaseStorage, n: int = 10):
    return storage.query(vector, n=n)


def in_memory_storage():
    return InMemoryStorage()


def in_memory_chroma_db_storage(collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.IN_MEMORY, collection_name=collection_name)


def persistent_chroma_db_storage(db_path: str, collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.PERSISTENT, db_path=db_path, collection_name=collection_name)


def server_client_chroma_db_storage(host: str, port: int, collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.SERVER_CLIENT, host=host, port=port, collection_name=collection_name)


__all__ = [
    'generate_spectrogram',
    'calc_prediction',
    'get_nearest',
    'in_memory_storage',
    'in_memory_chroma_db_storage',
    'persistent_chroma_db_storage',
    'server_client_chroma_db_storage',
]
