import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50

# User-configurable parameters
BATCH_SIZE = 32  # Batch size
EPOCHS = 50  # Number of epochs
DATA_FOLDER = r"C:\Users\coler\PycharmProjects\DeepLearning\Bird_Image_Training"  # Path to data folder
IMAGE_SIZE = (224, 224)  # Desired size of input images for the model
LEARNING_RATE = 0.005

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data augmentation and normalization transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load datasets from folders
train_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "test"), transform=transform)

# Create DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print data info for debugging
num_classes = len(train_dataset.classes)
print(f"Loaded {len(train_dataset)} training samples across {num_classes} classes.")
print(f"Loaded {len(val_dataset)} validation samples across {num_classes} classes.")
print(f"Loaded {len(test_dataset)} testing samples across {num_classes} classes.")

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=torch.arange(num_classes).numpy(),
    y=torch.tensor(train_dataset.targets).numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Initialize pretrained ResNet18 and make adjustments for the number of classes
model = resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # Add dropout to prevent overfitting
    nn.Linear(512, num_classes)
)
model = model.to(device)

# Define cross-entropy loss with class weights and Adam optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler to reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize TensorBoard to track training progress
writer = SummaryWriter(log_dir="runs/resnet18_training")

# Early stopping parameters
best_val_accuracy = 0.0
patience = 5
early_stop_counter = 0

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_dataloader):
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Forward pass
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        running_loss += loss.item()
        predicted_classes = torch.argmax(predictions, dim=1)
        correct_train += (predicted_classes == batch_labels).sum().item()
        total_train += batch_labels.size(0)

        # Print per-batch progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}, "
                  f"Loss: {loss.item():.4f}, Accuracy: {correct_train / total_train:.4f}")

    # Calculate epoch-level training accuracy and loss
    train_accuracy = correct_train / total_train
    train_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} Training Summary: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Log training metrics to TensorBoard
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_accuracy, epoch)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_dataloader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_predictions = model(val_inputs)
            loss = criterion(val_predictions, val_labels)

            running_val_loss += loss.item()
            predicted_classes = torch.argmax(val_predictions, dim=1)
            correct_val += (predicted_classes == val_labels).sum().item()
            total_val += val_labels.size(0)

    # Calculate validation accuracy and loss
    val_accuracy = correct_val / total_val
    val_loss = running_val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1} Validation Summary: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Log validation metrics to TensorBoard
    writer.add_scalar("Validation/Loss", val_loss, epoch)
    writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

    # Update learning rate using scheduler
    scheduler.step()

    # Early stopping logic
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stop_counter = 0
        print(f"New best model found! Saving model with Validation Accuracy: {best_val_accuracy:.4f}")
        torch.save(model.state_dict(), "best_model.pth")  # Save the best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Test phase
print("Starting evaluation on the test dataset...")
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for test_inputs, test_labels in test_dataloader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_predictions = model(test_inputs)
        loss = criterion(test_predictions, test_labels)

        test_loss += loss.item()
        predicted_classes = torch.argmax(test_predictions, dim=1)
        correct_test += (predicted_classes == test_labels).sum().item()
        total_test += test_labels.size(0)

# Calculate test accuracy and loss
test_accuracy = correct_test / total_test
test_loss = test_loss / len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Log test results to TensorBoard
writer.add_scalar("Test/Loss", test_loss)
writer.add_scalar("Test/Accuracy", test_accuracy)
writer.close()
