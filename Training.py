import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

#The two different locations of the files
input_folder = r'C:\Users\coler\OneDrive\Desktop\birds_train_small'
output_folder = r'C:\Users\coler\PycharmProjects\DeepLearning\Bird_Image_Training'

#splitting into 80% train, 10% test, 10% val
split_ratio = (0.8, 0.1, 0.1)

#random seed
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=1337,
    ratio=split_ratio,
    group_prefix=None
)

#image resize
img_size = (224, 224)
batch_size = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

#data augmentation for test data (only rescaling)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#data augmentation for validation data (only rescaling)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_dir = output_folder+'/train'
val_dir = output_folder+'/val'
test_dir = output_folder+'/test'

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

from keras.applications.resnet import ResNet50

#weights_path = r'C:\Users\coler\OneDrive\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_size[0], img_size[1], 3))




for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(504, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
)
with tf.device('/device:GPU:0'):
    model.fit(
        train_data,
        epochs=50,
        validation_data=val_data
)

model.save('/Users/colerutherford/PycharmProjects/DeepLearning/model.keras')


#evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)