import cv2
import os

name = input("Enter your name: ")
data_path = 'known_faces'
user_path = os.path.join(data_path, name)
os.makedirs(user_path, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{user_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Image {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) == 27 or count >= 50:  # ESC key or 50 images
        break

cap.release()
cv2.destroyAllWindows()
