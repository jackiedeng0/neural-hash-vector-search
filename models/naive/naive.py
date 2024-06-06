"""
    Naive Hasher -> Uses naive loss function to punish hashing algorithm that
    buckets dissimilar items together
"""
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

from models import HashingModel, array_to_binary, binary_to_array
from neighbor_table import NeighborTable

class NaiveHasher(HashingModel):
    def __init__(self, embeddings, hash_bits, summary=False):
        super().__init__(embeddings, hash_bits)
        # self.embeddings
        # self._embedding_dim
        # self.hash_bits
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self._embedding_dim,)),
                keras.layers.Dense(hash_bits, keras.activations.sigmoid),
            ]
        )

        if summary:
            self.model.summary()

        self.model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer=keras.optimizers.Adam(),
        )

    def hash_embedding(self, embedding):
        # Call the full model
        raw = self.model(embedding.reshape(1, self._embedding_dim))[0]
        # Binarize
        return array_to_binary(raw)

    def train(self, items, epochs=1, verbose=False):
        # Cache embeddings so we don't have to invoke self.embeddings again
        # every epoch
        train_embeddings = np.empty((len(items), self._embedding_dim))
        for i, t in enumerate(items):
            train_embeddings[i] = np.array(self.embeddings[t.lower()])

        for epoch in range(epochs):
            table = self.get_table(items)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                print("Current stats...")
                table.statistics_graph()
            # Calculate target output
            #
            # This is based on moving towards the direction of the mean
            # similarity of the words that the vector currently maps to. The
            # stronger the similarity is, the more we want to model to map the
            # current vector to its current hash. And conversely for negative
            # similarities, its magnitude determines how much we want to move
            # away from the current hash
            targets = np.empty((len(train_embeddings), self.hash_bits))
            average_mean_similarity = 0
            for i in range(len(train_embeddings)):
                embedding = train_embeddings[i]

                _hash = self.hash_embedding(embedding)
                mean_similarity = table.compare_to_bucket(_hash, embedding)
                average_mean_similarity += mean_similarity

                targets[i] = np.array(
                    binary_to_array(~_hash, self.hash_bits))
                targets[i] = ((targets[i] - 0.5) * mean_similarity) + 0.5
                targets[i].reshape(1, self.hash_bits)

            average_mean_similarity = (average_mean_similarity /
                                       len(train_embeddings))
            if verbose:
                print(
                f"Average Mean Similarity: {average_mean_similarity:.2f}")
                print("Training ...")

            # Train
            loss = self.model.train_on_batch(
                    np.array(train_embeddings), targets)
            if verbose:
                print(f"Loss: {loss:.2f}")
        
        if verbose:
            print("Training complete.")

