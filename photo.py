
import torch
from torchvision.transforms import ToPILImage
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
import os
from PIL import Image

# # Set NO_NNPACK environment variable to suppress NNPACK warning
os.environ["NO_NNPACK"] = "1"

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Initialize InceptionResnetV1 for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_cropped = mtcnn(img)
    return img_cropped

def get_embedding(img_tensor):
    with torch.no_grad():
        embedding = resnet(img_tensor.unsqueeze(0))
    return embedding

def cosine_similarity(embedding1, embedding2):
    return (embedding1 @ embedding2.T).item()

def recognize_face(img_path, known_embeddings, threshold=0.8):
    img_tensor = preprocess_image(img_path)
    
    if img_tensor is None:
        return None, "No face detected"
    
    embedding = get_embedding(img_tensor)
    max_similarity = -1
    matched_name = None
    
    for name, known_embedding in known_embeddings.items():
        similarity = cosine_similarity(embedding, known_embedding)
        if similarity > max_similarity and similarity >= threshold:
            max_similarity = similarity
            matched_name = name
    
    return matched_name, max_similarity

# Prepare dataset of known faces
known_faces_dir = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images'  # Directory containing known faces
known_face_image = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/srk.jpg'  # Path to the known face image

# Prepare dataset of known faces
known_embeddings = {}
name = 'khan_bhai'  # You can assign a name to the known face

img_tensor = preprocess_image(known_face_image)

if img_tensor is not None:
    embedding = get_embedding(img_tensor)
    known_embeddings[name] = embedding

# Define the test image path
test_image_path = '/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/bollywood.jpg'

# Add this code before the recognition process
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    image = transforms.ToPILImage()(tensor.cpu().detach())
    return image


# Display cropped faces from the test image
from torchvision.transforms import ToPILImage

# Display cropped faces from the test image
test_cropped_face = preprocess_image(test_image_path)
if test_cropped_face is not None:
    to_pil = ToPILImage()
    test_cropped_face_pil = to_pil(test_cropped_face)
    test_cropped_face_pil.show()
else:
    print("No face detected in the test image")

# Display cropped faces from known images
for name, embedding in known_embeddings.items():
    print("Known face:", name)
    known_cropped_face = preprocess_image(known_face_image)
    if known_cropped_face is not None:
        known_cropped_face_pil = to_pil(known_cropped_face)
        known_cropped_face_pil.show()
    else:
        print("No face detected in the known face image")


print("Processing test image:", test_image_path)

# Recognize a face from a new image
matched_name, similarity = recognize_face(test_image_path, known_embeddings)

if matched_name:
    print(f"Matched with {matched_name} (Similarity: {similarity:.2f})")
else:
    print("No match found or no face detected")
