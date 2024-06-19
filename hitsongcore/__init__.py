__version__ = '0.0.3'


from .file import Audio
from .model import ModelLoader
from .storages import BaseStorage, InMemoryStorage, ChromaDBStorage, ChromaDBMode


def generate_spectrogram(audio_file_path: str, is_force: bool = False):
    audio = Audio(audio_file_path)

    if not audio.exists_file():
        return False

    if not audio.exists_spectrogram() or is_force:
        audio.mkdir()
        audio.clear_all()
        audio.generate_spectrogram_slice()
    return True


def get_model_loader(device: str = None, model_path: str = 'model.pt', num_classes: int = 0):
    if device is None:
        device = 'cpu'
    return ModelLoader(device=device, model_path=model_path, num_classes=num_classes)


def get_in_memory_storage():
    return InMemoryStorage()


def get_in_memory_chroma_db_storage(collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.IN_MEMORY, collection_name=collection_name)


def get_persistent_chroma_db_storage(db_path: str, collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.PERSISTENT, db_path=db_path, collection_name=collection_name)


def get_server_client_chroma_db_storage(host: str, port: int, collection_name: str):
    return ChromaDBStorage(db_mode=ChromaDBMode.SERVER_CLIENT, host=host, port=port, collection_name=collection_name)


__all__ = [
    'generate_spectrogram',
    'get_model_loader',
    'get_in_memory_storage',
    'get_in_memory_chroma_db_storage',
    'get_persistent_chroma_db_storage',
    'get_server_client_chroma_db_storage',
]
