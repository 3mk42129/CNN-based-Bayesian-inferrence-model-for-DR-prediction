import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

BASE_DIR = r"C:\AOL AI\DR Dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "colored_images")
MODEL_PATH = os.path.join(BASE_DIR, "src", "cnn_model.h5")

CLASSES = ["Mild", "Moderate", "Severe"]

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    classes=CLASSES,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    classes=CLASSES,
    subset="validation"
)

print("Training Images:", train_gen.samples)
print("Validation Images:", val_gen.samples)
print("Total Images:", train_gen.samples + val_gen.samples)
print("Class Indices:", train_gen.class_indices)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

for layer in base_model.layers[:-40]:
    layer.trainable = False

for layer in base_model.layers[-40:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(len(CLASSES), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30
)

model.save(MODEL_PATH)

print("cnn saved")

final_train_acc = history.history["accuracy"][-1] * 100
final_val_acc = history.history["val_accuracy"][-1] * 100

labels = ["Training Accuracy", "Validation Accuracy"]
values = [final_train_acc, final_val_acc]

plt.figure()
plt.bar(labels, values)
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Overall CNN Model Accuracy")
plt.show()

val_gen.reset()

preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred, target_names=CLASSES))

# To activate type:
# & "C:\AOL AI\venv310\Scripts\Activate.ps1"