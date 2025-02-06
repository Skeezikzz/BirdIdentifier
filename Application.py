from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import torch
from torchvision import transforms
from load_and_predict import load_model, predict_image
from Bird_Dictionary_2 import class_to_name  # Assuming this contains the bird names

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the model once when the app starts
MODEL_PATH = "best_model.pth"  # Adjust the path to your model
NUM_CLASSES = len(class_to_name)  # Number of classes
model = load_model(MODEL_PATH, NUM_CLASSES)  # Load your trained model

# Define preprocessing (must match model training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route("/")
def home():
    """Render the homepage with the upload form."""
    return render_template("webpage.html")  # webpage.html contains the upload form


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, run prediction, and display result."""
    # Check if the POST request contains a file
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "No file selected", 400

    if file:  # If a file is uploaded
        # Save the file to the uploads folder
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Predict using your model
        try:
            predicted_class = predict_image(model, filepath, class_to_name, preprocess)
        except Exception as e:
            return f"Error during prediction: {str(e)}", 500

        # Render the results page with the prediction
        return render_template("results.html", bird_name=predicted_class, image_path=filepath)

    return "Something went wrong", 500


if __name__ == "__main__":
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask app
    app.run(debug=True)
