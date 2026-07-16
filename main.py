import os

from src.detector import FaceDetector
from src.clustering import FaceCluster
from src.confidence import calculate_confidence
from src.reporter import ReportGenerator


DATASET = "dataset"
OUTPUT = "output"


def main():

    detector = FaceDetector()

    embeddings = []

    image_paths = []

    print("\nExtracting Face Embeddings...\n")

    for image in sorted(os.listdir(DATASET)):

        if not image.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(DATASET, image)

        result = detector.get_embedding(path)

        if result is None:

            print(f"Skipped : {image}")

            continue

        embeddings.append(result["embedding"])

        image_paths.append(path)

        print(f"Processed : {image}")

    print("\nClustering Faces...\n")

    cluster_model = FaceCluster()

    labels = cluster_model.cluster(embeddings)

    confidence = calculate_confidence(
        embeddings,
        labels
    )

    report = ReportGenerator(OUTPUT)

    cluster_info = report.save_clusters(
        image_paths,
        labels
    )

    report.save_csv(
        image_paths,
        labels,
        confidence
    )

    report.save_json(cluster_info)

    print("\n============================")

    print("Completed Successfully")

    print("============================")

    print(f"\nImages Processed : {len(image_paths)}")

    print(f"Clusters Found : {len(set(labels))}")

    print("\nOutput Folder :", OUTPUT)


if __name__ == "__main__":
    main()