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

class BinarizationLayer(keras.layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        super(BinarizationLayer, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        return keras.ops.where(inputs > self.threshold, 1.0, 0.0)

    def get_config(self):
        config = super(BinarizationLayer, self).get_config()
        config.update({"threshold": self.threshold})
        return config

class BinaryAutoencoder(HashingModel):
    def __init__(self, embeddings, hash_bits, summary=False):
        super().__init__(embeddings, hash_bits)
        # self.embeddings
        # self._embedding_dim
        # self.hash_bits
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self._embedding_dim,)),
                # encoding
                keras.layers.Dense(hash_bits * 2, keras.activations.relu),
                keras.layers.Dense(hash_bits, keras.activations.sigmoid,
                                   name="EncoderDense2"),
                # decoding
                keras.layers.Dense(hash_bits * 2, keras.activations.relu),
                keras.layers.Dense(self._embedding_dim),
            ]
        )

        self.encoder = keras.Model(inputs=self.model.inputs,
                          outputs=
                               self.model.get_layer("EncoderDense2").output)

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
