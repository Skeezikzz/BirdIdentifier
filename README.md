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
- 

### **1. convert.JPEG.py**
#### **Description**
`convert.JPEG.py` is a utility script designed to process a dataset of images by converting them into standardized **JPEG format**. This ensures that all images in a dataset are consistent in format, making them compatible with downstream machine learning workflows like image classification. The script is especially useful when working with datasets that contain images in mixed or unsupported formats.
#### **Features**
1. **Directory File Processing**:
    - The script scans the specified folder structure, including subdirectories, and identifies all image files.

2. **Image Format Standardization**:
    - Converts images of various file formats (such as PNG, BMP, or TIFF) into the **JPEG** format.

3. **Image Quality Assurance**:
    - Optionally adjusts the quality of the converted images to save disk space without significantly degrading the image quality.

4. **Maintains Directory Structure**:
    - The converted images are saved in their respective directories, preserving the original folder hierarchy.

#### **Libraries Used**
1. **Pillow**:
    - Handles image file reading, processing, and conversion to the JPEG format.
    - Also supports resizing, quality adjustments, and various image manipulations.

2. **OS**:
    - Used to traverse directories and manage file paths during conversion.

3. **Shutil**:
    - Helps with file copying or moving in case of necessary file relocations.

  ### **2. BirdDictionary.py**
#### **Description**

`BirdDictionary.py` provides a **searchable bird dictionary** that serves as a knowledge base for bird species. It works by allowing users to query information about bird species, such as their common names, scientific names, descriptions, and other notable characteristics. This script could be integrated with the classification model as a reference tool to provide additional details about the classified species.
#### **Features**
1. **Predefined Bird Data**:
    - The script contains a dictionary or JSON-based database of bird species and their attributes.

2. **Integration with Classification**:
    - Can be used as an add-on to provide details about bird species identified by the classification model.

3. **Searchable by Species**:
    - Users can query the dictionary using common names or scientific names to retrieve information.

4. **Expandable**:
    - The bird dictionary can be expanded by adding more species or additional metadata (e.g., habitat, geographic range, migration patterns).

#### **Libraries Used**
1. **JSON** _(if using a JSON file as a dictionary)_:
    - Used to load, parse, and save the bird dictionary in a JSON format for better portability.

2. **OS** _(for managing file-based dictionaries)_:
    - Helps locate and load dictionary files dynamically.

3. (Optional) **argparse**:
    - Allows users to pass bird names via command line arguments for querying.
  
### **3. photoscraper.py**
#### **Description**
`photoscraper.py` is a web scraping tool designed to automate the process of downloading bird images from the internet. This script is particularly useful for researchers or machine learning enthusiasts who need to quickly gather visual datasets for training or testing purposes.
#### **Features**
1. **Automated Image Searching**:
    - Uses web scraping to search for bird images on platforms like Google Images or Bing.

2. **Bulk Image Downloading**:
    - Downloads a specified number of images for each bird species or keyword.

3. **Customizable Search Keywords**:
    - Users can specify the bird species names or other keywords as input.

4. **Image Validation**:
    - Ensures that the downloaded files are valid images (e.g., filters out non-image files).
    - Automatically skips duplicates.

5. **Output Organization**:
    - Saves images in structured folders corresponding to the search query (e.g., a folder for each bird species).

#### **Libraries Used**
1. **Requests**:
    - Sends HTTP requests to fetch webpages or image URLs.

2. **BeautifulSoup**:
    - Parses the downloaded HTML to extract image URLs.

3. **OS**:
    - Used to create output directories and manage file paths.

4. **Shutil**:
    - Assists with file management and organization.

5. **Pillow**:
    - Validates downloaded content to ensure that it's an image file.
  


