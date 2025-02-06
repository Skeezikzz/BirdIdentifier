import torch
from torchvision import transforms
from PIL import Image

# Import the bird class dictionary from bird_dictionary_2.py
from Bird_Dictionary_2 import class_to_name

# 1. Load the trained model
MODEL_PATH = "best_model.pth"  # Path to your trained model
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()  # Set the model to evaluation mode

# 2. Define the preprocessing transforms
# Resize image to match model's input size, normalize it for the model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])


def predict_image(image_path):
    """
    Predicts the bird class for an uploaded image.
    Args:
        image_path: Path to the uploaded image.

    Returns:
        Bird's scientific name (if found in `class_to_name`) or "Unknown Bird".
    """
    try:
        # 3. Open and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # 4. Perform inference with the model
        with torch.no_grad():
            output = model(image_tensor)  # Get model predictions
            _, predicted_idx = torch.max(output, 1)  # Get the index of the highest probability

        # 5. Map the predicted class index to the scientific name
        predicted_idx = predicted_idx.item()  # Convert tensor to integer
        bird_name = class_to_name.get(predicted_idx, "Unknown Bird")

        return bird_name

    except Exception as e:
        return f"Error during prediction: {str(e)}"


if __name__ == "__main__":
    # Example test
    image_path = r"C:\birds_train_small\03129_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Buteo_lineatus\0daf3c37-17c6-40d9-bfde-51644f7cce4f.jpg"  # Replace with the path of the test image
    prediction = predict_image(image_path)
    print(f"Predicted Bird: {prediction}")
