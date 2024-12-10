import os
import glob
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Path to dataset
file_path = 'dataset_blood_group'

# Validate dataset directory
assert os.path.exists(file_path), f"Dataset path '{file_path}' does not exist."

# List all classes
classes = os.listdir(file_path)
print(f"Classes found: {classes}")

# Extract file paths and labels
filepaths = glob.glob(file_path + '/**/*.bmp', recursive=True)  # Support .bmp files
labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]

# Create a dataframe for file paths and labels
data = pd.DataFrame({'Filepath': filepaths, 'Label': labels})

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.20, stratify=data['Label'], random_state=42)

# Visualize some images from the dataset (optional)
def visualize_sample_images(data, num_images=6):
    print("Visualizing sample images...")
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes.flat):
        img = plt.imread(data.iloc[i]['Filepath'])
        ax.imshow(img)
        ax.set_title(data.iloc[i]['Label'])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Uncomment to visualize images
# visualize_sample_images(data)

# Set up ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)

# Define the base pre-trained ResNet50 model
pretrained_model = ResNet50(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze the pre-trained layers
pretrained_model.trainable = False

# Add custom layers for classification
x = Dense(128, activation="relu")(pretrained_model.output)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(len(classes), activation='softmax')(x)

# Build the final model
model = Model(inputs=pretrained_model.input, outputs=outputs)

# Compile the model
model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=[early_stopping]
)

# Save the trained model
model_save_path = "model_blood_group_detection.h5"
print(f"Saving the model to {model_save_path}...")
model.save(model_save_path)
print("Model saved successfully!")

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(valid_gen)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Generate classification report
y_true = test['Label'].values
y_pred = model.predict(valid_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(valid_gen.class_indices.keys())
print("Classification Report:")
print(classification_report(valid_gen.classes, y_pred_classes, target_names=class_labels))
