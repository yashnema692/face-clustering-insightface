# Face Clustering using InsightFace

## Overview

This project automatically groups images of the same person using Face Recognition.

It detects the largest face in every image, extracts a facial embedding using InsightFace (ArcFace), clusters similar faces using DBSCAN, and generates confidence scores.

---

## Features

- Face Detection
- ArcFace Embedding Extraction
- Automatic Face Clustering
- Confidence Score Calculation
- CSV Report
- JSON Report
- Automatic Cluster Folder Creation

---

## Project Structure

```
dataset/
output/
src/
main.py
requirements.txt
```

---

## Libraries

- Python
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- Pandas
- scikit-learn

---

## Run

```bash
pip install -r requirements.txt

python main.py
```

---

## Output

```
output/

cluster_1/
cluster_2/
cluster_3/

results.csv
summary.json
```

---

## Algorithm

1. Read all images
2. Detect faces
3. Select largest face
4. Generate ArcFace embeddings
5. Cluster embeddings using DBSCAN
6. Calculate confidence scores
7. Generate reports
