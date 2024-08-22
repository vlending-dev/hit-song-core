from abc import abstractmethod
from enum import Enum

import chromadb
import torch


class ChromaDBMode(Enum):
    IN_MEMORY = 'in_memory'
    PERSISTENT = 'persistent'
    SERVER_CLIENT = 'server_client'


class BaseStorage:
    @abstractmethod
    def add(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def multi_add(self, data):
        raise NotImplementedError

    @abstractmethod
    def get(self, key):
        raise NotImplementedError

    @abstractmethod
    def query(self, vector, n=10):
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self):
        self._data = {}

    def add(self, key, value):
        self._data[key] = value

    def multi_add(self, data):
        self._data.update(data)

    def get(self, key):
        return self._data.get(key)

    @torch.no_grad()
    def query(self, vector, n=10):
        distance_key_list = []
        distance_value_list = []
        vector_tensor = torch.Tensor(vector)
        for _key, _value in self._data.items():
            _value_tensor = torch.Tensor(_value)
            cosine = torch.nn.CosineSimilarity(dim=vector_tensor.dim() - 1, eps=1e-6)
            distance = cosine(vector_tensor, _value_tensor)
            # distance = torch.sum(vector_tensor * _value_tensor) / (torch.sqrt(torch.sum(vector_tensor ** 2)) * torch.sqrt(torch.sum(_value_tensor ** 2)))
            distance_key_list.append(_key)
            distance_value_list.append(distance)

        distance_tensor = torch.Tensor(distance_value_list)
        n = min(n, len(distance_tensor))
        top_k = torch.topk(distance_tensor, n)
        result = dict()
        for _key, _value in zip(top_k.indices, top_k.values):
            result[distance_key_list[_key.item()]] = f'{_value.item():.4f}'
        return result

    def __len__(self):
        return len(self._data)

    def get_keys(self):
        return list(self._data.keys())


class ChromaDBStorage(BaseStorage):
    def __init__(self, db_mode: ChromaDBMode, db_path: str = None, host: str = None, port: int = None, collection_name: str = None):
        if db_mode == ChromaDBMode.IN_MEMORY:
            self._db = chromadb.Client()
        elif db_mode == ChromaDBMode.PERSISTENT:
            self._db = chromadb.PersistentClient(path=db_path)
        elif db_mode == ChromaDBMode.SERVER_CLIENT:
            self._db = chromadb.HttpClient(host=host, port=port)
        else:
            self._db = chromadb.Client()
        self._collection = self._db.get_or_create_collection(
            name=collection_name or 'default',
            metadata={
                'hnsw:space': 'cosine',
            },
        )

    @property
    def collection(self):
        return self._collection

    def add(self, key, vector):
        params = dict(
            ids=[key],
            embeddings=[vector],
        )
        self.collection.add(**params)

    def multi_add(self, data):
        keys = list(data.keys())
        vectors = list(data.values())
        params = dict(
            ids=keys,
            embeddings=vectors,
        )
        self.collection.add(**params)

    def get(self, key):
        include = ['embeddings']
        params = dict(
            ids=[key],
            include=include,
        )
        query_result = self.collection.get(**params)
        result_dict = dict()
        for _ids, _embeddings in zip(query_result['ids'], query_result['embeddings']):
            result_dict[_ids] = _embeddings
        return result_dict

    def query(self, vector, n: int = 10):
        include = ['distances']
        params = dict(
            query_embeddings=[vector],
            n_results=n,
            include=include,
        )
        query_result = self.collection.query(**params)
        result_dict = dict()
        for _ids, _distances in zip(query_result['ids'][0], query_result['distances'][0]):
            # default function : chromadb.utils.distance_functions.cosine
            result_dict[_ids] = f'{1 - _distances:.4f}'
        return result_dict

    def __len__(self):
        query_result = self.collection.get()
        return len(query_result['ids'])

    def get_keys(self):
        query_result = self.collection.get()
        return query_result['ids']
