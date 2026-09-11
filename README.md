# Facial Emotion Recognition with CNN

A computer-vision project that uses a trained Convolutional Neural Network to classify facial expressions into seven emotion classes and provides both real-time webcam inference and a lightweight Streamlit image interface.

## Features
- CNN-based facial-expression classification
- Seven classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 48x48 grayscale preprocessing
- Histogram equalization
- Haar Cascade face detection
- Temporal prediction smoothing in webcam mode
- Confidence display
- Streamlit image-upload demo
- Saved Keras model (model.json + model.h5)
- Original training notebook retained

## Architecture
The saved model is a Sequential CNN with stacked convolution, batch-normalization, ReLU, average-pooling and dropout blocks, followed by global average pooling and a 7-class softmax output.

Input shape: 48 x 48 x 1

## Project Structure
~~~text
facial-emotions-detections/
├── Facial_Emotion_Recognition_using_CNN.ipynb
├── model.json
├── model.h5
├── webcam.py
├── app.py
├── requirements.txt
└── README.md
~~~

## Run locally
~~~bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
~~~

### Streamlit demo
~~~bash
streamlit run app.py
~~~

Upload a JPG, JPEG, or PNG image containing a face.

### Webcam demo
~~~bash
python webcam.py
~~~
Press q to stop the webcam application.

## Model Pipeline
~~~text
Input Image
    ↓
Grayscale Conversion
    ↓
Haar Cascade Face Detection
    ↓
48 x 48 Resize
    ↓
Histogram Equalization
    ↓
Normalization
    ↓
CNN
    ↓
7-Class Softmax Prediction
~~~

## Reproducibility
The repository includes the original training notebook and exported model files used for inference. The notebook remains the source for the original training dataset, training configuration, and evaluation results.

## Limitations
- Facial-expression classification is not the same as reliably determining a person's internal emotional state.
- Performance can vary with lighting, pose, occlusion, image quality, and demographic characteristics.
- The displayed softmax value is model confidence, not a calibrated probability.
- This is an educational computer-vision demonstration, not a medical, psychological, hiring, or surveillance system.

## Skills Demonstrated
Python · TensorFlow/Keras · CNN · OpenCV · Computer Vision · Image Preprocessing · Model Inference · Streamlit · Real-time Webcam Processing

## Author
**Anurag Pareek**
