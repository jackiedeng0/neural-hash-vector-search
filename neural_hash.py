"""
    Data Preparation
"""

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

embeddings = KeyedVectors.load_word2vec_format("glove.6B.50d.txt",
                                               binary=False)

vocabulary = []
with open("vocabulary.test", "r") as f:
    vocabulary = f.read().splitlines()

import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(vocabulary,
                                                       test_size = 0.2,
                                                       random_state = 59874)

train_vectors = []
for t in train:
    train_vectors.append(embeddings[t.lower()])

test_vectors = []
for t in test:
    test_vectors.append(embeddings[t.lower()])

"""
    Model Creation
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras

# embedding shape
input_shape = (50,)
# output shape
hash_bits = 8

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(hash_bits, keras.activations.sigmoid)
    ]
)

model.summary()

model.compile(
    loss=keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(),
)

"""
    Hash Table Initialization

    Model outputs # hash_bits no. of floats from 0 to 1. When put together,
    and rounded we get a binary value e.g. 011011101. This then becomes the
    key for hash table entry.
"""

def floats_to_binary(l):
    binary = 0
    for val in l:
        binary = (binary << 1) + round(val)
    return binary

neighbor_table = dict()

for word, vec in zip(train, train_vectors):
    em = embeddings[word.lower()]
    y = model(em.reshape(1, 50))
    print(y.tolist()[0])
    hash_key = floats_to_binary(y.tolist()[0])
    print(hash_key)
    print(type(hash_key))
    if hash_key in neighbor_table:
        neighbor_table[hash_key].append(word)
    else:
        neighbor_table[hash_key] = [word]

print(neighbor_table)

"""
    Model Training
"""

