import os
import json
import shutil
import pandas as pd


class ReportGenerator:

    def __init__(self, output_folder):

        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)

    def save_clusters(self, image_paths, labels):

        cluster_paths = {}

        for img_path, label in zip(image_paths, labels):

            cluster_name = f"cluster_{label + 1}"

            cluster_folder = os.path.join(
                self.output_folder,
                cluster_name
            )

            os.makedirs(cluster_folder, exist_ok=True)

            shutil.copy2(
                img_path,
                os.path.join(
                    cluster_folder,
                    os.path.basename(img_path)
                )
            )

            cluster_paths.setdefault(cluster_name, []).append(
                os.path.basename(img_path)
            )

        return cluster_paths

    def save_csv(self, image_paths, labels, confidence):

        rows = []

        for img, label, score in zip(
            image_paths,
            labels,
            confidence
        ):

            rows.append({
                "Image": os.path.basename(img),
                "Cluster": f"cluster_{label + 1}",
                "Confidence (%)": score
            })

        df = pd.DataFrame(rows)

        csv_path = os.path.join(
            self.output_folder,
            "results.csv"
        )

        df.to_csv(csv_path, index=False)

    def save_json(self, cluster_paths):

        json_path = os.path.join(
            self.output_folder,
            "summary.json"
        )

        with open(json_path, "w") as file:

            json.dump(
                cluster_paths,
                file,
                indent=4
            )