import cv2
import os

def main():
    print("[INFO] Starting the face capture program...")

    name = input("Enter your name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        input("Press Enter to exit...")
        return

    # Create directory to save face images
    data_path = 'known_faces'
    user_path = os.path.join(data_path, name)
    os.makedirs(user_path, exist_ok=True)
    print(f"[INFO] Images will be saved to: {user_path}")

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        input("Press Enter to exit...")
        return

    # Load Haar cascade for face detection
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    max_images = 50

    print("[INFO] Face capture started. Press ESC to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("[INFO] No face detected in this frame.")
        
        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            file_path = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            print(f"[INFO] Saved image {count}: {file_path}")

            # Draw rectangle and image number
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show live video with rectangles
        cv2.imshow("Capturing Faces", frame)

        # ESC key or 50 images
        if cv2.waitKey(1) == 27 or count >= max_images:
            break

    print(f"[INFO] Finished capturing {count} images for {name}")
    cap.release()
    cv2.destroyAllWindows()
    input("Press Enter to exit...")  # Keeps command prompt window open

if __name__ == "__main__":
    main()
