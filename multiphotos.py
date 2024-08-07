import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load pre-trained FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Load and preprocess images
def load_image(image_path):
    img = Image.open(image_path)
    img = transforms.functional.to_tensor(img).to(device)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to recognize faces
def recognize_faces(image_paths):
    for image_path in image_paths:
        img = load_image(image_path)
        
        # Detect faces
        boxes, _ = mtcnn.detect(img)
        
        if boxes is not None:
            # Extract face embeddings
            embeddings = resnet(img)
            
            # Perform recognition
            
            # Here you would implement the recognition logic.
            # This could involve calculating distances between embeddings
            # of the detected faces and comparing them with embeddings 
            # of known faces in your database.
            
            # For simplicity, let's just print the detected boxes
            print("Detected faces in", image_path, len(boxes))
            for box in boxes:
                print("Box:", box)
        else:
            print("No faces detected in", image_path)

# Example usage
image_paths = ["/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/srk.jpg", "/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/priyanka.jpg", "/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/images/sushant.jpg"]  # Paths to your face images
recognize_faces(image_paths)
