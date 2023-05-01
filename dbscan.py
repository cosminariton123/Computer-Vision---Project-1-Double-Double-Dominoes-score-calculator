from sklearn.cluster import DBSCAN

def clusterize(data):
    model = DBSCAN(eps=90, min_samples=10**3,  n_jobs=-1).fit(data)
    return model.labels_