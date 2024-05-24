import numpy as np

"""
    Data Preparation
"""

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

embeddings = KeyedVectors.load_word2vec_format("glove.6B.50d.txt",
                                               binary=False)
EMBED_DIM = embeddings.vector_size

vocabulary = []
with open("vocabulary.test", "r") as f:
    vocabulary = f.read().splitlines()

import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(vocabulary,
                                                       test_size = 0.2,
                                                       random_state = 59874)

train_vectors = np.empty((len(train), EMBED_DIM))
for i, t in enumerate(train):
    train_vectors[i] = np.array(embeddings[t.lower()])

test_vectors = np.empty((len(test), EMBED_DIM))
for i, t in enumerate(test):
    test_vectors[i] = np.array(embeddings[t.lower()])

"""
    Model Creation
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

# embedding shape
input_shape = (EMBED_DIM,)
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
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
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
    y = model(vec.reshape(1, EMBED_DIM))
    hash_key = floats_to_binary(y.tolist()[0])
    if hash_key in neighbor_table:
        neighbor_table[hash_key].append(word)
    else:
        neighbor_table[hash_key] = [word]

import json

with open('untrained_table.test', 'w') as out:
    json.dump(neighbor_table, out, indent=2)

"""
    Model Training
"""

import numpy.linalg

def binary_to_floats(b, n):
    floats = []
    for i in range(n):
        floats.append(b & (1 << (n - i)))
    return floats

total_epochs = 30

for epoch in range(total_epochs):
    print(f"Epoch {epoch+1}/{total_epochs}")

    # Calculate target output
    #
    # This is based on moving towards the direction of the mean similarity
    # of the words that the vector currently maps to. The stronger the
    # similarity is, the more we want to model to map the current vector to
    # its current hash. And conversely for negative similarities, its
    # magnitude determines how much we want to move away from the current hash
    targets = np.empty((len(train_vectors), hash_bits))
    average_mean_similarity = 0
    for i in range(len(train_vectors)):
        vector = train_vectors[i]

        hash_key = floats_to_binary(model(vector.reshape(1, EMBED_DIM)).tolist()[0])
        neighbors = neighbor_table[hash_key]
        mean_similarity = 0
        for neighbor in neighbors:
            neighbor_vector = embeddings[neighbor.lower()]
            mean_similarity += numpy.dot(vector, neighbor_vector) / \
                (numpy.linalg.norm(vector) * numpy.linalg.norm(neighbor_vector))
        mean_similarity = mean_similarity / len(neighbors)
        average_mean_similarity += mean_similarity

        targets[i] = np.array(binary_to_floats(~hash_key, hash_bits))
        targets[i] = ((targets[i] - 0.5) * mean_similarity) + 0.5
        targets[i].reshape(1, hash_bits)

    average_mean_similarity = average_mean_similarity / len(train_vectors)
    print(f"Average Mean Similarity: {average_mean_similarity:.2f}")

    # Train
    print("Training ...")
    loss = model.train_on_batch(np.array(train_vectors), targets)
    print(f"Loss: {loss:.2f}")

    # With the new weights, recalculate what the new hash table looks like
    neighbor_table.clear()
    for word, vec in zip(train, train_vectors):
        y = model(vec.reshape(1, EMBED_DIM))
        hash_key = floats_to_binary(y.tolist()[0])
        if hash_key in neighbor_table:
            neighbor_table[hash_key].append(word)
        else:
            neighbor_table[hash_key] = [word]

"""
    Save Hash Table
"""

import json

with open('trained_table.test', 'w') as out:
    json.dump(neighbor_table, out, indent=2)

