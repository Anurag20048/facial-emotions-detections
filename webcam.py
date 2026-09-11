from collections import deque
import cv2
import numpy as np
from model_loader import load_model

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.equalizeHist(face)
    return (face.astype("float32") / 255.0).reshape(1, 48, 48, 1)

def main():
    model = load_model()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    buffer = deque(maxlen=7)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")
    print("Webcam started. Press q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
            for x, y, w, h in faces:
                prediction = model.predict(preprocess_face(gray[y:y+h, x:x+w]), verbose=0)[0]
                buffer.append(prediction)
                averaged = np.mean(buffer, axis=0)
                idx = int(np.argmax(averaged))
                label = f"{EMOTION_LABELS[idx]} ({averaged[idx] * 100:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            cv2.imshow("Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
