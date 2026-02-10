# Team AB3 – Realistic and Lightweight Driver Drowsiness Detection

## Team Info
- 22471A0538 — **Moghal Babar** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Problem formulation, system design, MobileNetV2 feature extraction, Logistic Regression modeling, experimentation, paper writing_

- 22471A0519 — **Gaddam raghu rami reddy** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Research guidance, methodology validation, result verification_

- 22471A0543 — **Nukala vinay ** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Dataset preparation, preprocessing, EDA, evaluation support_

- 22471A0549 — **Shaik Ismail** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Model testing, deployment assistance, documentation_

---

## Abstract
Driver drowsiness is one of the major contributors to road accidents worldwide, creating a strong need for accurate and real-time monitoring systems.  
This project presents a **realistic and lightweight driver drowsiness detection system** based on eye-state analysis. Deep visual features are extracted using **MobileNetV2**, a computationally efficient convolutional neural network optimized for embedded systems. Instead of using heavy end-to-end deep classifiers, the extracted features are classified using **Logistic Regression**, ensuring fast inference and interpretability.  
To enhance robustness under real-world conditions such as low illumination, motion blur, and sensor noise, **Gaussian noise augmentation** is applied during training. Experimental results on a balanced eye image dataset demonstrate **95% accuracy**, low inference latency (~8 ms per image), and a compact model size (~14 MB), making the system suitable for real-time deployment on edge and in-vehicle platforms.

---

## Paper Reference (Inspiration)
👉 **Realistic and Lightweight Driver Drowsiness Detection Using MobileNetV2 Features and Logistic Regression with Noise-Robust Learning**  
**Authors:** Moghal Babar, Dr. Sireesha Moturi, et al.  
Original IEEE-style research paper used as the foundation for this project.

---

## Our Improvement Over Existing Paper
- Adopted a **hybrid deep learning + classical machine learning** approach
- Used **MobileNetV2 strictly as a feature extractor**, reducing computational cost
- Introduced **Gaussian noise–augmented training** to improve real-world robustness
- Achieved a **compact model size (≈14 MB)** with competitive accuracy
- Optimized the system for **real-time inference on low-resource devices**

---

## About the Project
### What the project does
- Detects driver drowsiness by classifying eye images into **Opened** or **Closed** states

### Why it is useful
- Helps reduce fatigue-related road accidents
- Non-intrusive and camera-based
- Suitable for real-time driver monitoring systems

### General Workflow


---

## Dataset Used
👉 **Opened–Closed Eyes Dataset (Kaggle)**  
https://www.kaggle.com/datasets/hazemfahmy/openned-closed-eyes

### Dataset Details
- Total images: 4,103  
- Open-eye images: 2,171  
- Closed-eye images: 1,932  
- Balanced class distribution  
- Real-world variations in illumination, eye shape, and pose  

---

## Dependencies Used
- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  

---

## EDA & Preprocessing
- Dataset verification and class balance analysis
- Image resizing to **128 × 128**
- Pixel normalization
- Gaussian blur and noise augmentation
- Train–test split (80% training, 20% testing)
- Noise-aware preprocessing to enhance generalization

---

## Model Training Info
- Feature extractor: **Pre-trained MobileNetV2 (ImageNet)**
- Classification model: **Logistic Regression**
- Feature vector size: 62,720
- Training time: ~4.8 seconds
- Training environment: Google Colab (GPU)

---

## Model Testing / Evaluation
### Metrics Used
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Performance Results
- Accuracy: **95.0%**
- Precision: **0.94**
- Recall: **0.95**
- F1-score: **0.945**
- Inference time: **0.008 seconds per image**

---

## Results
- High recall for detecting **Closed-eye (drowsy)** states
- Robust performance under noisy and degraded visual conditions
- Lightweight and efficient inference suitable for real-time systems
- Better speed–accuracy trade-off compared to heavy CNN models

---

## Limitations & Future Work
### Limitations
- Uses only eye-state information
- Single-frame prediction without temporal context

### Future Work
- Incorporate blink rate and eye-closure duration
- Add head pose and yawning detection
- Extend to video-based temporal analysis
- Deploy on embedded platforms (Raspberry Pi, Jetson Nano)
- Integrate with in-vehicle driver assistance systems

---

## Deployment Info
- Flask-based web application
- Real-time image upload and prediction
- Optimized for edge and embedded deployment
- Suitable for vehicle driver monitoring platforms
