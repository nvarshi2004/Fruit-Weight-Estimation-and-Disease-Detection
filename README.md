# Fruit Weight Estimation and Disease Detection System 🍎

## Overview

This project is an intelligent computer vision system that predicts **apple weight** and detects **apple diseases** from images using Machine Learning and Deep Learning techniques.

The system takes an apple image as input, segments the fruit, extracts important features, and performs two tasks:

* **Weight Prediction** using machine learning regression models
* **Disease Detection** using deep learning CNN models

---

## Features

✅ Apple segmentation using YOLOv8
✅ Apple weight prediction
✅ Apple disease classification
✅ Flask web interface for user interaction
✅ Feature extraction and feature selection pipeline

---

## Tech Stack

### Programming Language

* Python

### Machine Learning / Deep Learning

* YOLOv8
* EfficientNetB0
* Linear Regression
* Random Forest
* Ridge Regression
* KNN
* XGBoost
* MLP

### Libraries

* OpenCV
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Matplotlib
* Ultralytics

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask

---

## Project Workflow

1. User uploads apple image
2. Image preprocessing
3. Apple segmentation using YOLOv8
4. Feature extraction
5. Weight prediction
6. Disease detection
7. Final output display

---

## Project Structure

```bash
apple_project/
│
├── Feature_extraction/
├── feature_selection/
├── training&testing/
├── HTML For Project/
│   ├── app.py
│   ├── index.html
│   ├── weightestimation.py
│   ├── Freshness prediction.py
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nvarshi2004/Fruit-Weight-Estimation-and-Disease-Detection.git
```

Move into project folder:

```bash
cd Fruit-Weight-Estimation-and-Disease-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
cd "HTML For Project"
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000/
```

---

## Dataset

Custom dataset created for:

* Apple images
* Weight labels
* Disease categories

Disease classes:

* Healthy
* Blotch
* Rot
* Scab

---

## Note

Large model files were excluded from GitHub due to file size limits:

* `yolov8x-seg.pt`
* `EfficientNet_Final.keras`

You can add them manually before running the project.

---

## Future Improvements

* Improve prediction accuracy
* Deploy on cloud
* Mobile application integration
* Real-time fruit grading system

---

## Author

**Nuggu Varshith**
B.Tech CSE (AI & ML)
GITAM University Bangalore
