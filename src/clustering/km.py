from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch
import matplotlib.pyplot as plt


def clustering_kmeans(data):
    m = Birch(branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True).fit(data.values)
    lbl = m.labels_
    plt.scatter(data[0], data[1], c=lbl)
    plt.show()
    print(lbl)
