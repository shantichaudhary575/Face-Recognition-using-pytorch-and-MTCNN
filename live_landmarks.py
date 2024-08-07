import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            if boxes is not None:
                print(f"Detected {len(boxes)} faces")
                for box, prob, ld in zip(boxes, probs, landmarks):
                    print(f"Box: {box}, Probability: {prob}")
                    # Draw rectangle on frame
                    cv2.rectangle(frame,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255),
                                  thickness=2)

                    # Show probability
                    cv2.putText(frame, str(
                        prob), (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    # Draw landmarks
                    for point in ld:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            else:
                print("No faces detected")
        except Exception as e:
            print(f"Error in drawing: {e}")
            pass

        return frame

    def run(self):
        """
        Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                frame = self._draw(frame, boxes, probs, landmarks)
            except Exception as e:
                print(f"Error in detection: {e}")
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the app
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()
