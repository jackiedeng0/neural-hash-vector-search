import json
import numpy as np
import matplotlib.pyplot as plt

class NeighborTable:
    def __init__(self, embeddings):
        self.table = dict()
        self._centroids = dict()
        self.embeddings = embeddings
        self._embedding_dim = embeddings.vector_size

    # Reused dict functions (as necessary)
    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, item):
        if key in self.table.keys():
            self.table[key].append(item)
            self._centroids[key] = (((self._centroids[key] *
                                     (len(self.table[key])-1)) + 
                                     self.embeddings[item.lower()]) /
                                    len(self.table[key]))
        else:
            self.table[key] = [item]
            self._centroids[key] = self.embeddings[item.lower()]

    def clear(self):
        self.table.clear()
        self._centroids.clear()

    def keys(self):
        return self.table.keys()

    def compare_to_bucket(self, key, vector):
        bucket = self.table[key]
        mean_similarity = 0
        for member in bucket:
            member_vector = self.embeddings[member.lower()]
            mean_similarity += np.dot(vector, member_vector) /            \
                (np.linalg.norm(vector) * np.linalg.norm(member_vector))
        mean_similarity /= len(bucket)
        return mean_similarity

    def mean_centroid_distance(self, key):
        bucket = self.table[key]
        bucket_vectors = np.empty((len(bucket), self._embedding_dim))
        for i, member in enumerate(bucket):
            bucket_vectors[i] = np.array(self.embeddings[member.lower()])
        centroid = self._centroids[key]
        return np.average(np.linalg.norm(centroid - bucket_vectors,
                                               axis=1))

    # Statistics for the distribution of items
    def statistics(self):
        stats = []
        for k in self.table.keys():
            stats.append((k, len(self.table[k]),
                         self.mean_centroid_distance(k)))
        return stats

    def statistics_graph(self):
        stats = self.statistics()
        keys, counts, mcds = [], [], []
        fig = plt.figure(figsize=(10, 8))
        ((axc, axm), (hc, hm)) = fig.subplots(2, 2)
        # Sorting for count and plotting
        stats = sorted(stats, key=lambda x: x[1])
        keys, counts, mcds = zip(*stats)
        # Make keys into strings so bars are graphed in order
        keys = list(map(str, keys))
        axc.bar(keys, counts)
        axc.set_title('Sorted by count')
        axc_m = axc.twinx()
        axc_m.plot(keys, mcds, 'x')

        # Sorting for mcd and plotting
        stats = sorted(stats, key=lambda x: x[2], reverse=True)
        keys, counts, mcds = zip(*stats)
        # Make keys into strings so bars are graphed in order
        keys = list(map(str, keys))
        axm.bar(keys, counts)
        axm.set_title('Sorted by mcd')
        axm_c = axm.twinx()
        axm_c.plot(keys, mcds, 'x')

        # Histograms
        amt, bins = np.histogram(counts)
        hc.set_title('Count histogram')
        hc.stairs(amt, bins)
        amt, bins = np.histogram(mcds)
        hm.set_title('MCD histogram')
        hm.stairs(amt, bins)

        # Show
        plt.show()

    def export_json(self, path):
        with open(path, 'w') as out:
            json.dump(self.table, out, indent=2)

