import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

# Load the trained model
model_path = '/Users/colerutherford/PycharmProjects/DeepLearning/model.keras'
model = load_model(model_path)

# Class names mapping (add your class-to-name mapping here)
class_names = {
    0: 'Cherry', 1: 'Coffee-plant', 2: 'Cucumber', 3: 'Fox_Nut(Makhana)', 4: 'Lemon',
    5: 'Olive-Tree', 6: 'Pearl-Millet(Bajra)', 7: 'Tobacco Plant', 8: 'Almond', 9: 'Banana',
    10: 'Cardamom', 11: 'Chilli', 12: 'Clove', 13: 'Coconut', 14: 'Cotton', 15: 'Gram',
    16: 'Jowar', 17: 'Jute', 18: 'Maize', 19: 'Mustard-Oil', 20: 'Papaya', 21: 'Pineapple',
    22: 'Rice', 23: 'Soyabean', 24: 'Sugarcane', 25: 'Sunflower', 26: 'Tea', 27: 'Tomato',
    28: 'Vigna-Radiata(Mung)', 29: 'Wheat'
}


# Function to preprocess and predict for a single image
def predict_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    img = cv2.resize(img, (224, 224))
    # Convert image to numpy array and expand dimensions for batch
    img_array = np.expand_dims(img, axis=0)
    # Preprocess the image (same as during training)
    img_array = preprocess_input(img_array)

    # Run prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, predictions[0][predicted_class_index]


# Example usage
image_path = '/Users/colerutherford/Downloads/owdijcweo.png'  # Replace with your image path
predicted_class, confidence = predict_image(image_path)
print(f"Predicted Class: {predicted_class} with Confidence: {confidence:.2f}")
