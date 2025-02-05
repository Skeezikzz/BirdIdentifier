import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Set device for training (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================
# 1. Data Loading and Preprocessing
# ==============================================================
# Path to the dataset folder
dataset_root = "path_to_dataset_folder"  # Replace with your dataset folder path

# Define data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of pre-trained model
    transforms.RandomHorizontalFlip(),  # Data augmentation (horizontal flips)
    transforms.RandomRotation(10),  # Slight random rotations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=f"{dataset_root}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f"{dataset_root}/val", transform=val_test_transforms)
test_dataset = datasets.ImageFolder(root=f"{dataset_root}/test", transform=val_test_transforms)

# Handle class imbalance by computing class weights
class_counts = np.bincount([label for _, label in train_dataset.samples])
class_weights = compute_class_weight("balanced", classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Push class weights to GPU

# DataLoaders
batch_size = 64  # Adjust depending on available GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")

# ==============================================================
# 2. Model Architecture (Fine-Tuning ResNet50)
# ==============================================================
# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Update final fully connected layer to match the number of classes (1500)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Output size = 1500 classes
model = model.to(device)

# Define the loss function (with class weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (reduces learning rate on plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

# ==============================================================
# 3. Training Loop
# ==============================================================
num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 40)

    # Training phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples
    print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(preds == labels).item()
            val_total_samples += labels.size(0)

    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_accuracy = val_correct_predictions / val_total_samples
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

    # Save the best model
    if val_epoch_accuracy > best_val_accuracy:
        best_val_accuracy = val_epoch_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

    # Update learning rate
    scheduler.step(val_epoch_loss)

# ==============================================================
# 4. Testing the Model
# ==============================================================
print("\nTesting the model on the test dataset...")
model.eval()
test_correct_predictions = 0
total_test_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct_predictions += torch.sum(preds == labels).item()
        total_test_samples += labels.size(0)

test_accuracy = test_correct_predictions / total_test_samples
print(f"\nTest Accuracy: {test_accuracy:.4f}")