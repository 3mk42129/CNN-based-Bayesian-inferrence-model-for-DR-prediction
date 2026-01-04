import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from bayesian import infer_risk
from sklearn.metrics import r2_score

BASE_DIR = r"C:\AOL AI\DR Dataset"
CNN_MODEL_PATH = os.path.join(BASE_DIR, "src", "cnn_model.h5")
IMAGE_DIR = os.path.join(BASE_DIR, "colored_images", "Test images")
PATIENT_CSV = os.path.join(BASE_DIR, "patients.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "prediction_results.csv")

cnn_model = load_model(CNN_MODEL_PATH)

class_names = ["Mild", "Moderate", "Severe"]
severity_values = np.array([1, 2, 3])

# Load patient data
patient_data = {}
with open(PATIENT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        patient_data[row["image"]] = {
            "age": int(row["age"]),
            "duration": float(row["diabetic_duration"])
        }

# Only read test images that have patient data
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
    and f in patient_data
]

total = len(image_files)
count = 1

with open(OUTPUT_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image",
        "Age",
        "Diabetes_Duration",
        "Predicted_Class",
        "Regression_Severity",
        "Regression_R2",
        "Final_Risk"
    ])

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)

        age = patient_data[img_name]["age"]
        duration_years = patient_data[img_name]["duration"]

        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        cnn_probs = cnn_model.predict(x, verbose=0)[0][:3]
        cnn_probs = cnn_probs / np.sum(cnn_probs)

        regression_severity = np.dot(cnn_probs, severity_values)

        idx = int(round(regression_severity)) - 1
        idx = max(0, min(idx, 2))
        predicted_class = class_names[idx]

        
        risk_probs = infer_risk(
            cnn_probs,
            age=age,
            duration_years=duration_years
        )
        final_risk = ["Low", "Medium", "High"][int(np.argmax(risk_probs))]

        coef = np.polyfit(severity_values, cnn_probs, 1)
        y_fit = coef[0] * severity_values + coef[1]
        r2 = r2_score(cnn_probs, y_fit)

        print(f"[{count}/{total}] {img_name}")
        print("Age:", age)
        print("Diabetes Duration:", duration_years)
        print("CNN Probabilities:", cnn_probs)
        print("Regression Severity:", round(regression_severity, 2))
        print("Regression R²:", round(r2, 3))
        print("Predicted Class:", predicted_class)
        print("Final Risk:", final_risk)
        print("-" * 60)

        writer.writerow([
            img_name,
            age,
            duration_years,
            predicted_class,
            round(regression_severity, 2),
            round(r2, 3),
            final_risk
        ])

        # Plot
        x_line = np.linspace(1, 3, 100)
        y_line = coef[0] * x_line + coef[1]

        plt.figure(figsize=(6,4))
        plt.plot(severity_values, cnn_probs, "o-", linewidth=2, label="CNN Probabilities")
        plt.plot(x_line, y_line, "--", linewidth=2, label="Linear Regression")
        plt.axvline(regression_severity, linestyle=":", linewidth=2, label="Regression Severity")

        plt.xticks(severity_values, class_names)
        plt.xlabel("DR Severity")
        plt.ylabel("Probability")
        plt.title(f"{img_name}   (R²={round(r2,3)})")
        plt.legend()
        plt.grid(True)
        plt.show()

        count += 1

print("======================================")
print("TOTAL IMAGES:", total)
print("======================================")