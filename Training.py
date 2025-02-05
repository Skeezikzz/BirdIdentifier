import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# The two different locations of the files
input_folder = r'C:\birds_train_small'
output_folder = r'C:\Users\coler\PycharmProjects\DeepLearning\Bird_Image_Training'

# Splitting into 80% train, 10% test, 10% val
split_ratio = (0.8, 0.1, 0.1)
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=1337,
    ratio=split_ratio,
    group_prefix=None
)

# Image resizing
img_size = (299, 299)
batch_size = 64

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,  # ResNet50 preprocessing
    rotation_range=50,  # Increased rotation
    width_shift_range=0.4,  # Improved width shifts
    height_shift_range=0.4,  # Improved height shifts
    shear_range=0.4,  # Increased shear
    zoom_range=0.5,  # Larger zoom range
    channel_shift_range=25.0,  # Adjust color channels
    horizontal_flip=True,  # Flipping as usual
    fill_mode='nearest'  # Filling empty pixels after transformations
)

# Data generators for validation and testing (only rescaling)
val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

# Data directories
train_dir = output_folder + '/train'
val_dir = output_folder + '/val'
test_dir = output_folder + '/test'

# Loading image data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Compute class weights for handling class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Load and modify the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze all layers except the last 10 layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Enable training for the last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Build the overall model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Squeeze the spatial dimensions
    layers.Dense(256, activation='relu'),  # Intermediate layer with 256 neurons
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(train_data.num_classes, activation='softmax')  # Output layer with softmax
])

# Compile the model with a reduced learning rate to avoid catastrophic forgetting
model.compile(
    optimizer=Adam(learning_rate=4e-4),  # Fine-tuning with lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
)

# Set up a learning rate scheduler to decrease learning rate if validation loss plateaus
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,  # Wait for 3 epochs of no improvement
    min_lr=1e-6
)

# Train the model
with tf.device('/device:GPU:0'):
    model.fit(
        train_data,
        epochs=50,
        validation_data=val_data,
        callbacks=[lr_schedule],  # Learning rate scheduler
        class_weight=class_weights  # Apply class weights to balance poorly represented classes
    )

# Save the model
model.save('/Users/colerutherford/PycharmProjects/DeepLearning/model_finetuned.keras')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)
