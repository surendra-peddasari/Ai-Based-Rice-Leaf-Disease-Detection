import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = os.path.join("rice_leaf_dataset", "Data")
IMG_SIZE = (150, 150)
BATCH_SIZE = 8

# Validation Generator without shuffle to align ground truth labels
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load saved model
model = tf.keras.models.load_model("rice_leaf_model.h5")

# Predict on validation data
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

class_labels = list(val_generator.class_indices.keys())

# Print Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\n--- Confusion Matrix Array ---")
print(cm)
print("\nClasses ordered:", class_labels)