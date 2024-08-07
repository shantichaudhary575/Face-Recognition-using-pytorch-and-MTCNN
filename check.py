# import os
# import cv2
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from pathlib import Path

# class FaceDetector(object):
#     def __init__(self, mtcnn, resnet, known_face_embeddings, known_face_names, threshold=1.0):
#         self.mtcnn = mtcnn
#         self.resnet = resnet
#         self.known_face_embeddings = known_face_embeddings
#         self.known_face_names = known_face_names
#         self.threshold = threshold

#     def _draw(self, frame, boxes, probs, landmarks, names):
#         try:
#             if boxes is not None:
#                 for box, prob, ld, name in zip(boxes, probs, landmarks, names):
#                     # Draw rectangle on frame
#                     cv2.rectangle(frame,
#                                   (int(box[0]), int(box[1])),
#                                   (int(box[2]), int(box[3])),
#                                   (0, 0, 255),
#                                   thickness=2)

#                     # Show probability and name
#                     cv2.putText(frame, f'{name} ({prob:.2f})', (int(box[0]), int(box[1]) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

#                     # Draw landmarks
#                     for point in ld:
#                         cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
#             else:
#                 print("No faces detected")
#         except Exception as e:
#             print(f"Error in drawing: {e}")
#             pass

#         return frame

#     def recognize_faces(self, frame, boxes):
#         if boxes is None:
#             return []

#         faces = []
#         for box in boxes:
#             face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
#             face = cv2.resize(face, (160, 160))
#             face = np.transpose(face, (2, 0, 1)) / 255.0
#             faces.append(torch.tensor(face).float())

#         if len(faces) == 0:
#             print("No faces detected in the current frame.")
#             return []

#         faces = torch.stack(faces).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         embeddings = self.resnet(faces).detach().cpu()

#         names = []
#         for embedding in embeddings:
#             distances = [(name, torch.dist(embedding, known_embedding).item()) for name, known_embedding in zip(self.known_face_names, self.known_face_embeddings)]
#             name, distance = min(distances, key=lambda x: x[1])
#             if distance < self.threshold:
#                 names.append(name)
#             else:
#                 names.append("Unknown")

#         return names

#     def run(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print("Error: Could not open video.")
#             return

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             try:
#                 # detect face box, probability and landmarks
#                 boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
#                 # recognize faces
#                 names = self.recognize_faces(frame, boxes)
#                 # draw on frame
#                 frame = self._draw(frame, boxes, probs, landmarks, names)
#                 # Print recognized names for this frame
#                 for name in names:
#                     print(f"Recognized face: {name}")
#             except Exception as e:
#                 print(f"Error in detection: {e}")
#                 pass

#             # Show the frame
#             cv2.imshow('Face Detection and Recognition', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# # Load known face embeddings and names from saved images in a directory
# def load_known_faces_from_directory(directory):
#     known_face_names = []
#     known_face_embeddings = []

#     resnet = InceptionResnetV1(pretrained='vggface2').eval()

#     if not Path(directory).exists() or not Path(directory).is_dir():
#         print(f"Directory '{directory}' does not exist or is not a directory.")
#         return [], []

#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             if filename.endswith(".jpg"):
#                 filepath = os.path.join(dirpath, filename)
                
#                 img = cv2.imread(filepath)
#                 if img is None:
#                     print(f"Failed to read image: {filepath}")
#                     continue
                
#                 img = cv2.resize(img, (160, 160))
#                 img = np.transpose(img, (2, 0, 1)) / 255.0
#                 img_tensor = torch.tensor(img).unsqueeze(0).float()
                
#                 with torch.no_grad():
#                     embedding = resnet(img_tensor).squeeze().cpu()
                
#                 known_face_names.append(os.path.basename(dirpath))  
#                 known_face_embeddings.append(embedding)

#     if not known_face_names:
#         print("Warning: No face images found in the specified directory.")
#         return [], []

#     print(f"Loaded {len(known_face_names)} known faces.")
#     print(f"First known face name: {known_face_names[0]}")
#     print(f"First known face embedding shape: {known_face_embeddings[0].shape}")

#     return known_face_names, known_face_embeddings

# def extract_faces_from_video(train_video_path, output_dir, max_frames=200):
#     mtcnn = MTCNN(keep_all=True)
#     cap = cv2.VideoCapture(train_video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {train_video_path}.")
#         return
#     base_name = os.path.splitext(os.path.basename(train_video_path))[0]
#     output_dir = os.path.join(output_dir, base_name) 
#     os.makedirs(output_dir, exist_ok=True)
#     frame_count = 0
#     while cap.isOpened() and frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

#         if boxes is not None:
#             for i, box in enumerate(boxes):
#                 face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
#                 face_filename = os.path.join(output_dir, f'{base_name}_{frame_count}_{i}.jpg')
#                 cv2.imwrite(face_filename, face)
        
#         frame_count += 1

#     cap.release()

# train_video_path = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/shanti.mp4'
# output_dir = 'dataset/train'
    
# extract_faces_from_video(train_video_path, output_dir, max_frames=200)

# known_face_names, known_face_embeddings = load_known_faces_from_directory(output_dir)

# known_face_embeddings = torch.stack(known_face_embeddings)

# mtcnn = MTCNN()
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# fcd = FaceDetector(mtcnn, resnet, known_face_embeddings, known_face_names, threshold=0.8)

# test_video_path = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/no_shanti1.mp4'
# fcd.run(test_video_path)


import torch
print(torch.backends.mps.is_available())