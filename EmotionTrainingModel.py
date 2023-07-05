import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

# Set the paths for the training and validation image data directories
train_dir = 'images/train'
val_dir = 'images/validation'

# Set up the ImageDataGenerators for the training and validation datasets
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Set up the generators for the training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Set up the input layer for the convolutional neural network model
input_layer = Input(shape=(48,48,1))

# Add convolutional and pooling layers to the model
conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_layer)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
bn1 = BatchNormalization()(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
drop1 = Dropout(0.25)(pool1)

conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(drop1)
bn2 = BatchNormalization()(conv3)
conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu')(bn2)
bn3 = BatchNormalization()(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(bn3)
drop2 = Dropout(0.25)(pool2)

conv5 = Conv2D(256, kernel_size=(3, 3), activation='relu')(drop2)
bn4 = BatchNormalization()(conv5)
conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu')(bn4)
bn5 = BatchNormalization()(conv6)
pool3 = MaxPooling2D(pool_size=(2, 2))(bn5)
drop3 = Dropout(0.25)(pool3)

# Flatten the output of the final convolutional layer
flat1 = Flatten()(drop3)

# Add fully connected layers to the model
dense1 = Dense(512, activation='relu')(flat1)
bn6 = BatchNormalization()(dense1)
drop4 = Dropout(0.5)(bn6)

output_layer = Dense(7, activation='softmax')(drop4)

# Create the model
emotion_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with an optimizer, loss function, and metrics
emotion_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on the training data and validate on the validation data

emotion_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

emotion_model.save('emotion_model2.h5')