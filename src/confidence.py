import numpy as np


def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


def calculate_confidence(embeddings, labels):

    confidence = []

    for index, embedding in enumerate(embeddings):

        cluster = labels[index]

        cluster_embeddings = [
            embeddings[i]
            for i in range(len(labels))
            if labels[i] == cluster
        ]

        centroid = np.mean(cluster_embeddings, axis=0)

        score = cosine_similarity(
            embedding,
            centroid
        )

        confidence.append(round(float(score * 100), 2))

    return confidence