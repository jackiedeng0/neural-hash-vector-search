import numpy as np

from abc import ABC, abstractmethod
from neighbor_table import NeighborTable

# Utils for conversion between list of floats (vector representation) and
# binary (dict key)
def array_to_binary(a):
    binary = 0
    for val in a:
        binary = (binary << 1) + round(val)
    return int(binary)

def binary_to_array(b, n):
    a = np.empty(n)
    for i in range(n):
        a[i] = 1 if (b & (1 << (n - 1 - i))) else 0
    return a

# Base Hashing Model
class HashingModel(ABC):
    def __init__(self, embeddings, hash_bits):
        self.embeddings = embeddings
        self._embedding_dim = embeddings.vector_size
        self.hash_bits = hash_bits

    def get_table(self, items):
        # Returns a table that the current hasher would produce for the input
        # items
        table = NeighborTable(self.embeddings)
        for item in items:
            # Currently, items are words and our embedding model only takes
            # lower case words
            _embedding = self.embeddings[item.lower()]
            _hash = self.hash_embedding(_embedding)
            table[_hash] = item
        return table

    @abstractmethod
    def hash_embedding(self, embedding):
        # Returns the hash for a given embedding
        pass

    @abstractmethod
    def train(self, items, epochs=1, verbose=False):
        # Self-supervised training for the hasher to learn the input items
        pass
