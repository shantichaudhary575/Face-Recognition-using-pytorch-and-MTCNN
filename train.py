import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a dataset instance
dataset = datasets.ImageFolder(root='/Users/thapahemmagar/Documents/ord/Mtcnn_face_detection/dataset/train', transform=transform)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define a CNN model
# Define a CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define layers (convolutional, pooling, fully connected)
    
    def forward(self, x):
        # Define forward pass
        pass  # Placeholder for the forward pass implementation

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
    # Validate the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Generate YAML file
info = {
    'model_name': 'CNN',
    'input_size': [3, 224, 224],  # Assuming input image size is 224x224 with 3 channels
    'output_size': len(dataset.classes),  # Number of output classes
    'num_epochs': num_epochs,
    'accuracy': accuracy,
}

with open('model_info.yml', 'w') as file:
    yaml.dump(info, file)
