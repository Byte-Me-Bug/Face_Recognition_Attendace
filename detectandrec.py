import cv2
import numpy as np
import pickle

# Load DNN Face Detection Model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load trained recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", "rb") as f:
    label_dict = pickle.load(f)

# Flip label_dict to get name from label
id_to_name = label_dict
# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face_roi_color = frame[y1:y2, x1:x2]

            try:
                gray_face = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray_face, (100, 100))  # LBPH works best with fixed size

                label, conf = recognizer.predict(resized)

                name = id_to_name.get(label, "Unknown")
                text = f"{name} ({int(conf)})" if conf < 100 else "Unknown"

                color = (0, 255, 0) if conf < 100 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f"[WARN] Recognition error: {e}")
                continue

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
