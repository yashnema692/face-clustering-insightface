import numpy as np
from sklearn.cluster import DBSCAN


class FaceCluster:

    def __init__(self):

        self.model = DBSCAN(
            eps=0.6,
            min_samples=1,
            metric="cosine"
        )

    def cluster(self, embeddings):

        embeddings = np.array(embeddings)

        labels = self.model.fit_predict(embeddings)

        return labels