# Deep Learning Bird Classification Project
This project implements a **deep learning-based bird species classification system** using **TensorFlow** and **Keras** frameworks. The goal of this project is to classify bird images into their respective species by leveraging a pre-trained **ResNet50** model. The dataset is organized into training, validation, and testing sets, which are then preprocessed and used to fine-tune the deep learning model for accurate predictions.
## Features of the Project
1. **Dataset Preparation and Splitting:**
    - Bird image dataset is located in a folder containing subdirectories for each bird species.
    - **SplitFolders** is used to split the dataset into **training**, **validation**, and **testing** sets with an **80/10/10 ratio**.

2. **Preprocessing and Augmentation:**
    - Images are resized to **224x224 pixels** to match the input shape required by the ResNet50 model.
    - Extensive data augmentation is performed on the training set using the **ImageDataGenerator** class to improve performance and reduce overfitting.

3. **Transfer Learning with ResNet50:**
    - The project uses a **pre-trained ResNet50** model (with ImageNet weights) for feature extraction, freezing its convolutional base layers.
    - The classification head is tailored to fit the number of bird species (dynamically determined from the dataset).

4. **Model Training and Evaluation:**
    - The model is compiled with:
        - Optimizer: **Adam**
        - Loss Function: **Categorical Crossentropy**
        - Metric: **Accuracy**

    - Training is performed for **25 epochs**, including validation at each epoch.
    - The trained model is evaluated on the test dataset to compute overall accuracy.

5. **Model Save and Deployment:**
    - After training, the model is saved in TensorFlow's `keras` format (`.keras`) so that it can be loaded and used for inference in future applications.

## Libraries and Tools Used
### 1. **TensorFlow/Keras**
- **TensorFlow and Keras:** The primary deep learning framework used to build and train the model.
- Used to define the model architecture (ResNet50), train the model, and save the trained model for deployment.

### 2. **SplitFolders**
- Automates the splitting of the dataset into training, validation, and testing sets with a given ratio (80%, 10%, 10%).
- Handles randomizing the data split while keeping the class distribution consistent across sets.

### 3. **ImageDataGenerator**
- Essential for preprocessing and augmenting the dataset.
- The training set undergoes transformations such as:
    - **Rotation**: Rotates images randomly within ±20 degrees.
    - **Width/Height Shifts**: Adjusts images horizontally and vertically by up to 20% of their dimensions.
    - **Shearing and Zooming**: Randomly shears and zooms the images for diversity.
    - **Horizontal Flip**: Flips images horizontally as part of augmentation.

- Ensures that validation and test sets are only normalized without augmentation for unbiased evaluation.

### 4. **ResNet50**
- Pre-trained convolutional neural network model used for transfer learning.
- ResNet50 significantly reduces training time and improves performance by leveraging pre-learned features from the ImageNet dataset.

### 5. **Matplotlib**
- Used for visualizing training and validation results, including plots for **accuracy** and **loss** trends over epochs.

### 6. **NumPy**
- Used for numerical computations and matrix operations during preprocessing.

### 7. **Lxml and Pillow**
- **Lxml** is used internally for XML parsing if required.
- **Pillow** is used for image manipulation (like resizing) within the `ImageDataGenerator`.
