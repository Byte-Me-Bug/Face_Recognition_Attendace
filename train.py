import cv2
import os
import numpy as np
import pickle

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths
dataset_path = "known_faces"
label_dict = {}  # {label_id: name}
face_samples = []
face_labels = []
current_id = 0

# Loop through each person's folder
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing {person_name}")
    label_dict[current_id] = person_name

    for img_name in os.listdir(person_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
                face_samples.append(face_roi)
                face_labels.append(current_id)
                break  # Only one face per image

    current_id += 1

# Train the recognizer
print("[INFO] Training recognizer...")
recognizer.train(face_samples, np.array(face_labels))

# Save the trained model
recognizer.save("trainer.yml")

# Save the label mapping
with open("labels.pickle", "wb") as f:
    pickle.dump(label_dict, f)

print("[INFO] Training complete. Model saved as 'trainer.yml'.")
