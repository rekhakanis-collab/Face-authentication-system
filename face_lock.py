import cv2
import os
import time

# Load face detection model
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Prepare face recognizer and directories
if not os.path.exists("model"):
    os.makedirs("model")

recognizer = cv2.face.LBPHFaceRecognizer_create()
labels_file = "model/labels.txt"
trainer_file = "model/trainer.yml"

# Instruction messages
print("===============================================")
print("      FACE DETECTION SECURITY LOCK SYSTEM      ")
print("===============================================")
print("1. Press 'A' to register as ADMIN")
print("2. Press 'U' to register as USER")
print("3. Press 'R' to start RECOGNITION (unlock system)")
print("4. Press 'ESC' to EXIT")
print("===============================================")

def capture_faces(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found. Exiting.")
        return

    count = 0
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)
    print(f"Capturing faces for {name}. Please look at the camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{save_path}/{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Registering Face", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{count} face images saved for {name}.")

def train_model():
    faces = []
    ids = []
    label_map = {}
    current_id = 0
    dataset_path = "dataset"

    if not os.path.exists(dataset_path):
        print("No dataset found. Please register a face first.")
        return

    for name in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, name)
        if not os.path.isdir(folder):
            continue
        label_map[current_id] = name
        for image_file in os.listdir(folder):
            img_path = os.path.join(folder, image_file)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(gray)
            ids.append(current_id)
        current_id += 1

    recognizer.train(faces, np.array(ids))
    recognizer.save(trainer_file)

    with open(labels_file, "w") as f:
        for id_, name in label_map.items():
            f.write(f"{id_}:{name}\n")

    print("Model training completed.")

def recognize_face():
    if not os.path.exists(trainer_file) or not os.path.exists(labels_file):
        print("No trained data found. Please register and train first.")
        return

    recognizer.read(trainer_file)
    label_map = {}
    with open(labels_file, "r") as f:
        for line in f:
            if ":" in line:
                id_, name = line.strip().split(":", 1)
                label_map[int(id_)] = name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found. Exiting.")
        return

    print("Face recognition started. Press ESC to exit.")
    UNLOCK_CONFIDENCE_THRESHOLD = 60
    last_unlock = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            label_id, confidence = recognizer.predict(face_resized)
            name = label_map.get(label_id, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if confidence < UNLOCK_CONFIDENCE_THRESHOLD:
                if name == "admin":
                    print(f"Access Granted: {name} recognized (confidence={int(confidence)})")
                    print("Door Unlocked (Admin Access)")
                else:
                    print(f"Access Denied: {name} is not an admin.")
                time.sleep(3)
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

import numpy as np

while True:
    choice = input("Enter your choice (A/U/R/ESC): ").strip().lower()
    if choice == 'a':
        name = "admin"
        capture_faces(name)
        train_model()
    elif choice == 'u':
        name = input("Enter your name: ").strip()
        capture_faces(name)
        train_model()
    elif choice == 'r':
        recognize_face()
    elif choice == 'esc':
        print("Exiting the system.")
        break
    else:
        print("Invalid choice. Please enter A, U, R or ESC.")
