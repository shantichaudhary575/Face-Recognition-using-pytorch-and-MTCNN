
from facenet_pytorch import MTCNN
import cv2
import numpy as np

# Initialize MTCNN
detector = MTCNN()

# Load the image using OpenCV
img = cv2.imread('/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/bollywood.jpg')

# Convert the image from BGR (OpenCV default) to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces
boxes, probs, landmarks = detector.detect(img_rgb, landmarks=True)

# Ensure boxes, probs, and landmarks are not None
if boxes is not None and landmarks is not None:
    for i in range(len(boxes)):
        box = boxes[i]
        landmark = landmarks[i]
        
        x, y, width, height = int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])
        
        left_eye_x, left_eye_y = int(landmark[0][0]), int(landmark[0][1])
        right_eye_x, right_eye_y = int(landmark[1][0]), int(landmark[1][1])
        nose_x, nose_y = int(landmark[2][0]), int(landmark[2][1])
        mouth_left_x, mouth_left_y = int(landmark[3][0]), int(landmark[3][1])
        mouth_right_x, mouth_right_y = int(landmark[4][0]), int(landmark[4][1])

        cv2.circle(img, center=(left_eye_x, left_eye_y), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(right_eye_x, right_eye_y), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(nose_x, nose_y), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(mouth_left_x, mouth_left_y), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(mouth_right_x, mouth_right_y), color=(255, 0, 0), thickness=3, radius=2)

        cv2.rectangle(img, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)

    cv2.imshow('window', img)
    cv2.waitKey(0)
else:
    print("No faces detected.")
