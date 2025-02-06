import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from Bird_Dictionary_2 import bird_scientific_names

# Step 1: Function to load the ResNet model with the pre-trained state_dict
def load_model(model_path, num_classes):
    """
    Loads the ResNet model with the pre-trained weights.

    Args:
        model_path (str): Path to the saved model (.pth file).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Loaded model ready for inference.
    """
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=False)  # Change if a different ResNet variant was used
    # Update the classifier head for the correct number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load the model's state_dict
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Set the model to evaluation mode
    model.eval()

    return model


# Step 2: Function to make predictions
def predict_image(model, image_path, class_to_name, preprocess):
    """
    Predicts the class of a given image.

    Args:
        model (torch.nn.Module): The loaded model.
        image_path (str): Path to the image file.
        class_to_name (dict): Mapping of class indices to names.
        preprocess (torchvision.transforms.Compose): Preprocessing pipeline.

    Returns:
        str: Predicted class name.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)  # Get model predictions
            _, predicted_idx = torch.max(output, 1)  # Get the class index

        # Convert the class index to the class name
        predicted_idx = predicted_idx.item()  # Convert Tensor to int
        try:
            predicted_name = class_to_name[predicted_idx]
        except IndexError:
            predicted_name = "Unknown Class"

        return predicted_name

    except Exception as e:
        return f"Error: {e}"


# Step 3: Main script
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "best_model.pth"  # Path to your saved state_dict
    IMAGE_PATH = r"C:\birds_train_small\03126_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Buteo_buteo\0dac3557-1bd0-4e89-96d5-57303de0ca66.jpg"  # Path to the image file for inference
    NUM_CLASSES = 1486  # Number of classes used in training, update this
    class_to_name = {}  # Update with your own class index to name mapping

    # Preprocessing (must match training preprocessing)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize input image (adjust if needed)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = load_model(MODEL_PATH, NUM_CLASSES)

    # Predict the class for an image
    predicted_class = predict_image(model, IMAGE_PATH, bird_scientific_names, preprocess)

    print(f"Predicted Class: {predicted_class}")
