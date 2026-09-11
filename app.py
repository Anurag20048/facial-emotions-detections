from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from model_loader import load_model

BASE_DIR = Path(__file__).resolve().parent
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict(model, gray_face):
    face = cv2.resize(gray_face, (48, 48))
    face = cv2.equalizeHist(face).astype("float32") / 255.0
    probabilities = model.predict(face.reshape(1, 48, 48, 1), verbose=0)[0]
    idx = int(np.argmax(probabilities))
    return LABELS[idx], float(probabilities[idx])

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="centered")
st.title("🙂 Facial Emotion Recognition")
st.caption("CNN-based facial-expression classification with image upload.")
st.info("Predictions describe facial-expression classes from the model. They should not be treated as a reliable measure of a person's internal emotional state.")

try:
    model = get_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector().detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
    if len(faces) == 0:
        st.warning("No face was detected. Try a clearer, front-facing image.")
    else:
        output = frame.copy()
        results = []
        for x, y, w, h in faces:
            label, confidence = predict(model, gray[y:y+h, x:x+w])
            results.append((label, confidence))
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, f"{label} ({confidence*100:.1f}%)", (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        st.image(output, caption="Model prediction", use_container_width=True)
        for i, (label, confidence) in enumerate(results, 1):
            st.metric(f"Face {i}", label, f"{confidence*100:.1f}% confidence")
