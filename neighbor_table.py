import json
import numpy
import numpy.linalg
import matplotlib.pyplot as plt

# Utils for conversion between list of floats (vector representation) and
# binary (dict key)
def array_to_binary(a):
    binary = 0
    for val in a:
        binary = (binary << 1) + round(val)
    return binary

def binary_to_array(b, n):
    a = numpy.empty(n)
    for i in range(n):
        a[i] = 1 if (b & (1 << (n - 1 - i))) else 0
    return a

class NeighborTable:
    def __init__(self, embeddings):
        self.table = dict()
        self._centroids = dict()
        self.embeddings = embeddings
        self._embed_dim = embeddings.vector_size

    # Reused dict functions (as necessary)
    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, item):
        self.table[key] = item

    def clear(self):
        self.table.clear()
        self._centroids.clear()

    def keys(self):
        return self.table.keys()

    # Functions prefixed with an 'a_' take an array as input that gets
    # transformed into a dict key
    def a_get(self, array):
        return self.table[array_to_binary(array)]

    def a_set(self, array, word):
        key = array_to_binary(array)
        if key in self.table.keys():
            self.table[key].append(word)
            self._centroids[key] = (((self._centroids[key] *
                                     (len(self.table[key])-1)) + 
                                     self.embeddings[word.lower()]) /
                                    len(self.table[key]))
        else:
            self.table[key] = [word]
            self._centroids[key] = self.embeddings[word.lower()]

    def a_compare_to_bucket(self, array, vector):
        bucket = self.table[array_to_binary(array)]
        mean_similarity = 0
        for member in bucket:
            member_vector = self.embeddings[member.lower()]
            mean_similarity += numpy.dot(vector, member_vector) /            \
                (numpy.linalg.norm(vector) * numpy.linalg.norm(member_vector))
        mean_similarity /= len(bucket)
        return mean_similarity

    def mean_centroid_distance(self, key):
        bucket = self.table[key]
        bucket_vectors = numpy.empty((len(bucket), self._embed_dim))
        for i, member in enumerate(bucket):
            bucket_vectors[i] = numpy.array(self.embeddings[member.lower()])
        centroid = self._centroids[key]
        return numpy.average(numpy.linalg.norm(centroid - bucket_vectors,
                                               axis=1))

    # Statistics for the distribution of items
    def statistics(self):
        keys = [*self.table]
        counts = []
        mean_centroid_distances = []
        for k in self.table.keys():
            counts.append(len(self.table[k]))
            mean_centroid_distances.append(self.mean_centroid_distance(k))
        return keys, counts, mean_centroid_distances

    def statistics_graph(self):
        keys, counts, mcds = self.statistics()
        fig, ax = plt.subplots()
        ax.bar(keys, counts)
        plt.show()

    def export_json(self, path):
        with open(path, 'w') as out:
            json.dump(self.table, out, indent=2)

