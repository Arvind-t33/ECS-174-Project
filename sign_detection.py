import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image

# Data paths
data_dir = 'gtrsb/Train'
csv_file = 'gtrsb/Train/Train.csv'

# Load and preprocess the data
def load_data(data_dir, csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in data.iterrows():
        img_path = os.path.join(data_dir, row['ClassId'], row['Path'])
        image = Image.open(img_path)
        image = image.resize((32, 32))
        images.append(np.array(image))
        labels.append(row['ClassId'])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = load_data(data_dir, csv_file)
images = images / 255.0
labels = to_categorical(labels, num_classes=43)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

datagen.fit(X_train)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')