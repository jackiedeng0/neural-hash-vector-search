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

import numpy.linalg

def binary_to_floats(b, n):
    floats = []
    for i in range(n):
        floats.append(b & (1 << (n - i)))
    return floats

"""
total_epochs = 10

for epoch in range(total_epochs):
    print(f"Epoch {epoch+1}/{total_epochs}")
"""

print("Current weights:")
print(model.get_weights())
#for i in range(len(train_vectors)):
for i in range(5):
    vector = np.array([train_vectors[i]])

    # Calculate target output
    #
    # This is based on moving towards the direction of the mean similarity
    # of the words that the vector currently maps to. The stronger the
    # similarity is, the more we want to model to map the current vector to
    # its current hash. And conversely for negative similarities, its
    # magnitude determines how much we want to move away from the current hash
    hash_key = floats_to_binary(model(vector).tolist()[0])
    neighbors = neighbor_table[hash_key]
    mean_similarity = 0
    for neighbor in neighbors:
        neighbor_vector = embeddings[neighbor.lower()]
        mean_similarity += numpy.dot(vector, neighbor_vector) / \
            (numpy.linalg.norm(vector) * numpy.linalg.norm(neighbor_vector))
    mean_similarity = mean_similarity / len(neighbors)

    target = np.array(binary_to_floats(~hash_key, hash_bits))
    target = ((target - 0.5) * mean_similarity) + 0.5
    target.reshape(1, hash_bits)

    model.train_on_batch(np.array([train_vectors[i]]), target)

    print(f"After {i}")
    print(model.get_weights())
