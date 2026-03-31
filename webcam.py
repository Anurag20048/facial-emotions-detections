import cv2
import numpy as np
from keras.models import model_from_json
from collections import deque


with open("model.json", "r") as f:
    model = model_from_json(f.read())

model.load_weights("model.h5")


emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


emotion_buffer = deque(maxlen=7)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Webcam started. Press q to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.equalizeHist(face)   # improves lighting robustness
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)


        prediction = model.predict(face, verbose=0)[0]
        emotion_buffer.append(prediction)

        avg_prediction = np.mean(emotion_buffer, axis=0)
        emotion_index = np.argmax(avg_prediction)
        emotion = emotion_labels[emotion_index]
        confidence = avg_prediction[emotion_index] * 100


        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
