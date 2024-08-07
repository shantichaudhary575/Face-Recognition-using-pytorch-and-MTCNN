import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path

class FaceDetector(object):
    def __init__(self, mtcnn, resnet, known_face_embeddings, known_face_names):
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.known_face_embeddings = known_face_embeddings
        self.known_face_names = known_face_names

    def _draw(self, frame, boxes, probs, landmarks, names, confidences):
        if boxes is not None:
            for box, prob, ld, name, confidence in zip(boxes, probs, landmarks, names, confidences):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)
                cv2.putText(frame, f'{name} ({prob:.2f}, {confidence:.2f})', (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                for point in ld:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        else:
            print("No faces detected")
        return frame

    def recognize_faces(self, frame, boxes):
        if boxes is None:
            return [], []

        faces = []
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue  # Skip if the box is out of the frame bounds
            
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue  # Skip if the face region is empty
            
            face = cv2.resize(face, (160, 160))
            face = np.transpose(face, (2, 0, 1)) / 255.0
            faces.append(torch.tensor(face).float())

        if len(faces) == 0:
            print("No faces detected in the current frame.")
            return [], []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet.to(device)
        faces = torch.stack(faces).to(device)
        embeddings = self.resnet(faces).detach().cpu()

        names = []
        confidences = []
        for embedding in embeddings:
            distances = [(name, torch.dist(embedding, known_embedding).item()) for name, known_embedding in zip(self.known_face_names, self.known_face_embeddings)]
            name, distance = min(distances, key=lambda x: x[1])
            confidence = 1 / (1 + distance)
            if distance < 0.65:  # Threshold for face recognition
                names.append(name)
                confidences.append(confidence)
            else:
                names.append("Unknown")
                confidences.append(confidence)

        return names, confidences

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        detected_faces = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_no = 0
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
            names, confidences = self.recognize_faces(frame, boxes)
            frame = self._draw(frame, boxes, probs, landmarks, names, confidences)
            frame_no += 1

            for i, (name, confidence) in enumerate(zip(names, confidences)):
                if confidence > 0.65:  # Confidence threshold
                    print(f"Recognized face: {name}, {confidence:.2f}, {frame_no}, {frame_no / fps}, {boxes[i]}")
                    if name != "Unknown" and name not in detected_faces:
                        detected_faces.append(name)

            cv2.imshow('Face Detection and Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(detected_faces)

def load_known_faces_from_directory(directory):
    known_face_names = []
    known_face_embeddings = []

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    if not Path(directory).exists() or not Path(directory).is_dir():
        print(f"Directory '{directory}' does not exist or is not a directory.")
        return [], []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".jpg"):
                filepath = os.path.join(dirpath, filename)
                img = cv2.imread(filepath)
                if img is None:
                    print(f"Failed to read image: {filepath}")
                    continue

                img = cv2.resize(img, (160, 160))
                img = np.transpose(img, (2, 0, 1)) / 255.0
                img_tensor = torch.tensor(img).unsqueeze(0).float()

                with torch.no_grad():
                    embedding = resnet(img_tensor).squeeze().cpu()

                known_face_names.append(os.path.basename(dirpath))
                known_face_embeddings.append(embedding)

    if not known_face_names:
        print("Warning: No face images found in the specified directory.")
        return [], []

    return known_face_names, known_face_embeddings

def train(train_video_path, name):
    max_frames = 200
    output_dir = 'dataset/train'
    mtcnn = MTCNN(keep_all=True)
    cap = cv2.VideoCapture(train_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {train_video_path}.")
        return

    base_name = name
    output_dir = os.path.join(output_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue  # Skip if the box is out of the frame bounds
                
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue  # Skip if the face region is empty
                
                face_filename = os.path.join(output_dir, f'{base_name}_{frame_count}_{i}.jpg')
                cv2.imwrite(face_filename, face)

        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}")

def ensure_training(train_video_path, output_dir, name):
    face_dir = os.path.join(output_dir, name)
    if not os.path.exists(face_dir) or not os.listdir(face_dir):
        train(train_video_path, name)
    else:
        print("Dataset already exists, skipping training.")

# Paths to videos and directories
train_video_path = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/samir.mp4'
output_dir = 'dataset/train'
name = "samir"

# Ensure training is done if needed
ensure_training(train_video_path, output_dir, name)

# Load known faces from the directory
known_face_names, known_face_embeddings = load_known_faces_from_directory(output_dir)
known_face_embeddings = torch.stack(known_face_embeddings)

# Initialize FaceDetector with the MTCNN and Resnet models
mtcnn = MTCNN(keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.8])
resnet = InceptionResnetV1(pretrained='vggface2').eval()
fcd = FaceDetector(mtcnn, resnet, known_face_embeddings, known_face_names)

test_video_path = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/samirtest.mp4'

# Function to run the face detection and recognition
def test(video_path):
    fcd.run(video_path)

# Run the test
test(test_video_path)
