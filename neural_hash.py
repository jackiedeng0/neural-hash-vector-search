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

import neighbor_table

table = neighbor_table.NeighborTable(embeddings)

for word, vec in zip(train, train_vectors):
    y = model(vec.reshape(1, EMBED_DIM))
    table.a_set(np.array(y[0]), word)

table.export_json("untrained_table.test")

"""
    Model Training
"""

import numpy.linalg

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

        y = np.array(model(vector.reshape(1,EMBED_DIM))[0])
        mean_similarity = table.a_compare_to_bucket(y, vector)
        hash_key = neighbor_table.array_to_binary(y)
        average_mean_similarity += mean_similarity

        targets[i] = np.array(neighbor_table.binary_to_array(~hash_key, hash_bits))
        targets[i] = ((targets[i] - 0.5) * mean_similarity) + 0.5
        targets[i].reshape(1, hash_bits)

    average_mean_similarity = average_mean_similarity / len(train_vectors)
    print(f"Average Mean Similarity: {average_mean_similarity:.2f}")

    # Train
    print("Training ...")
    loss = model.train_on_batch(np.array(train_vectors), targets)
    print(f"Loss: {loss:.2f}")
    print("Statistics:")
    print(table.statistics())

    # With the new weights, recalculate what the new hash table looks like
    table.clear()
    for word, vec in zip(train, train_vectors):
        y = model(vec.reshape(1, EMBED_DIM))
        table.a_set(np.array(y[0]), word)

"""
    Save Hash Table
"""

table.export_json("trained_table.test")

