from sklearn.model_selection import train_test_split

SPLIT_RANDOM_SEED = 83186

def load_train_test_split(vocabulary_file,
                          embeddings,
                          test_size = 0.2):
    vocabulary = []
    with open(vocabulary_file, "r") as f:
        vocabulary = f.read().splitlines()
    train, test = train_test_split(vocabulary,
                                   test_size = test_size,
                                   random_state = SPLIT_RANDOM_SEED)

    return train, test
