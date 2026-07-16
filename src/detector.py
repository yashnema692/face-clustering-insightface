import cv2
from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(self):

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def get_embedding(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return None

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        # Select the largest detected face
        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) *
                             (face.bbox[3] - face.bbox[1])
        )

        return {
            "embedding": largest_face.embedding,
            "bbox": largest_face.bbox,
            "det_score": float(largest_face.det_score)
        }