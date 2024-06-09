"""
    Data Preparation
"""

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

embeddings = KeyedVectors.load_word2vec_format("embeddings/lexvec.enwiki+newscrawl.300d.W.pos.vectors",
                                               binary=False)


from data_loader.vocab import load_train_test_split

training_set, test_set = \
    load_train_test_split(vocabulary_file="data/test-vocabulary.txt",
                          embeddings=embeddings)


"""
    Model Creation
"""

from models.binary_autoencoder import BinaryAutoencoder

HASH_BITS=32
hasher = BinaryAutoencoder(embeddings, hash_bits=HASH_BITS, summary=True)
table = hasher.get_table(training_set)
table.statistics_graph()
table.export_json("results/untrained_table.test")


"""
    Model Training
"""

TOTAL_EPOCHS=200
hasher.train(training_set, epochs=TOTAL_EPOCHS, verbose=True)
table = hasher.get_table(training_set)
table.statistics_graph()
table.export_json("results/trained_table.test")

