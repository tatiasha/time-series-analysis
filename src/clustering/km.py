from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch
import matplotlib.pyplot as plt


def clustering_kmeans(data):
    # m = Birch(branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True).fit(data.values)
    m = KMeans(n_clusters=10).fit(data.values)
    lbl = m.labels_
    print("CLUSTERING - {} clusters".format(len(set(lbl))))
    plt.scatter(data[0], data[1], c=lbl)
    plt.show()
    print(lbl)
