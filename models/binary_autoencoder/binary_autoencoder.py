"""
    Binary autoencoder -> Encoder network produces binarized hash key and
    decoder network reconstructs vectors from the hash key. We train on the
    sum of squared residuals between the input and reconstructed vectors.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

from models import HashingModel, array_to_binary, binary_to_array
from neighbor_table import NeighborTable

QUANTIZATION_THRESHOLD = 0.5

class QuantizationLossLayer(keras.layers.Layer):
    def __init__(self, weighting):
        super().__init__()
        self.weighting = weighting

    def call(self, inputs):
        self.add_loss(self.weighting *
            keras.ops.average(
                keras.ops.abs(
                    inputs - 
                    keras.ops.where(
                        inputs > QUANTIZATION_THRESHOLD, 1.0, 0.0)
                    )))
        return inputs

class BinaryAutoencoder(HashingModel):
    def __init__(self, embeddings, hash_bits, summary=False):
        super().__init__(embeddings, hash_bits)
        # self.embeddings
        # self._embedding_dim
        # self.hash_bits
        inputs = keras.Input(shape=(self._embedding_dim,))
        # encoding
        encode1 = keras.layers.Dense(hash_bits * 2, keras.activations.relu)(inputs)
        encode2 = keras.layers.Dense(hash_bits, keras.activations.sigmoid,
                           name="EncoderDense2")(encode1)
        qloss = QuantizationLossLayer(weighting=0.01)(encode2)
        # decoding
        decode1 = keras.layers.Dense(hash_bits * 2, keras.activations.relu)(qloss)
        decode2 = keras.layers.Dense(self._embedding_dim)(decode1)

        self.model = keras.Model(inputs=inputs, outputs=decode2)

        self.encoder = keras.Model(inputs=inputs, outputs=encode2)

        if summary:
            self.model.summary()

        self.model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer=keras.optimizers.Adam(),
        )

    def hash_embedding(self, embedding):
        # Call encoding part of model
        raw = self.encoder(embedding.reshape(1, self._embedding_dim))[0]
        # Binarize
        return array_to_binary(raw)

    def train(self, items, epochs=1, verbose=False):
        # Cache embeddings so we don't have to invoke self.embeddings again
        # every epoch
        train_embeddings = np.empty((len(items), self._embedding_dim))
        for i, t in enumerate(items):
            train_embeddings[i] = np.array(self.embeddings[t.lower()])

        callback = keras.callbacks.EarlyStopping(monitor='loss')

        self.model.fit(train_embeddings,
                       train_embeddings,
                       epochs=epochs,
                       callbacks=[callback])
