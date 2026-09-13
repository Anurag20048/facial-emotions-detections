# Facial Emotion Recognition with CNN

> A computer-vision application that classifies facial expressions into seven emotion categories using a trained Convolutional Neural Network.

## 🚀 Overview
This project demonstrates an end-to-end facial-expression recognition pipeline from image preprocessing and face detection to CNN inference. It provides both a Streamlit image interface and real-time webcam prediction.

## 📸 Demo
Use the Streamlit interface to upload an image and view the predicted expression category with the model confidence.

## ✨ Features
- CNN-based facial-expression classification
- Seven emotion categories
- 48×48 grayscale preprocessing
- Histogram equalization
- Haar Cascade face detection
- Temporal smoothing for webcam predictions
- Confidence display
- Streamlit image interface
- Real-time webcam inference
- Exported Keras model
- Original training notebook

## 🛠️ Tech Stack
Python · TensorFlow/Keras · OpenCV · NumPy · Streamlit · CNN · Haar Cascade

## 📦 Installation
```bash
git clone https://github.com/Anurag20048/facial-emotions-detections.git
cd facial-emotions-detections
python -m venv .venv
```

Windows:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ▶️ Usage
```bash
streamlit run app.py
```

For webcam inference:
```bash
python webcam.py
```

## 📁 Project Structure
```text
facial-emotions-detections/
├── Facial_Emotion_Recognition_using_CNN.ipynb
├── model.json
├── model.h5
├── webcam.py
├── app.py
├── requirements.txt
└── README.md
```

## 🔧 Configuration
The exported model files are included and loaded by the inference applications.

## 🧪 Running Tests
The primary validation workflow is inference through the supplied Streamlit and webcam applications.

## 🗺️ Roadmap
- [ ] Improve model evaluation and calibration
- [ ] Add preprocessing experiments
- [ ] Add lightweight deployment

## 🤝 Contributing
Pull requests are welcome. For major changes, open an issue first.

## 📄 License
See the `LICENSE` file.

## 👤 Author
**Anurag Pareek**
- GitHub: https://github.com/Anurag20048
