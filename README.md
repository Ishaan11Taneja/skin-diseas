


# 🧠 SkinCareAI – AI-Powered Skin Disease Diagnosis System

SkinCareAI is a machine learning-based web application that detects and classifies various skin diseases from dermatoscopic images. Built with Python, scikit-learn, TensorFlow, and Flask, it enables users to upload images and receive instant, severity-based predictions through a modern and responsive web interface.

---

## 📌 Features

- 🔼 **Image Upload:** Drag-and-drop or select files directly in the browser
- ⚡ **Real-Time Analysis:** Instant classification and severity assessment
- 🧾 **9 Skin Conditions Covered**, grouped by severity:

  **High Severity**
  - Melanoma  
  - Squamous Cell Carcinoma  

  **Medium Severity**
  - Actinic Keratosis  
  - Atopic Dermatitis  
  - Tinea / Ringworm / Candidiasis  

  **Low Severity**
  - Benign Keratosis  
  - Dermatofibroma  
  - Melanocytic Nevus  
  - Vascular Lesion  

- 📊 **Probability Scores** for each condition
- ✅ **Severity Classification** (Low / Medium / High)
- 📈 **Performance Metrics Dashboard** with accuracy, precision, recall, F1-score, and confusion matrix
- 🌐 **Responsive UI** using HTML, CSS, JavaScript

---

## 🧠 Model Performance

| Model            | Validation Accuracy |
|------------------|---------------------|
| **SimpleCNN**        | **86%**             |
| Random Forest    | 85%                 |
| InceptionV3      | 60%                 |
| VGG16            | 22%                 |
| ResNet50         | 10%                 |
| EfficientNetB0   | 8%                  |

✅ **SimpleCNN** and **Random Forest** performed best, proving that simpler, well-optimized models can outperform deeper architectures on specialized datasets.

---

## 📂 Project Structure


skin-disease-detection/
├── static/              # CSS, JS files
├── templates/           # HTML template
├── train/ & val/        # Image datasets
├── uploads/             # Uploaded image storage
├── app.py               # Flask web app
├── train\_model.py       # Model training script
├── download\_kaggle\_dataset.py
├── download\_images.py
├── skin\_disease\_model.joblib
├── class\_names.txt
└── requirements.txt


---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.8+
- pip

### 📦 Setup Instructions

bash
git clone <repository-url>
cd skin-disease-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


---

## 🚀 Usage

### 1️⃣ Train the Model

bash
python train_model.py


This will:

* Load and preprocess the training data
* Train the model (SimpleCNN, InceptionV3, etc.)
* Save the trained model and class labels

### 2️⃣ Launch the Web App

bash
python app.py


Open your browser and visit:


http://localhost:5000


Upload an image and click **"Analyze Image"** to receive a prediction with severity.

---

## 🗃 Dataset

* Source: **DermNet + HAM10000**
* \~19,500 images (training & testing)
* JPEG, RGB format
* Preprocessing:

  * Resizing to 299x299 px
  * Normalization (scaled to \[0,1])
  * Augmentation: rotation, flip, zoom

---

## 📊 Evaluation Metrics

* ✅ Accuracy
* 🎯 Precision
* 📢 Recall
* 📏 F1-Score
* 🔲 Confusion Matrix

---

## 🔮 Future Scope

* 📷 Expand dataset diversity (age, ethnicity, lighting)
* 🔎 Add explainable AI (Grad-CAM, LIME)
* 🏥 Collaborate with dermatologists for clinical validation
* ☁️ Cloud API integration for real-time predictions

---

## 🤝 Contributing

We welcome contributions!
If you'd like to help improve this project, feel free to fork the repo, submit issues, or create pull requests.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* 🧠 Supervised by: **Prof. Vishakha Arya**
* 🏫 Institution: **DIT University, Dehradun**
* 📚 Datasets: [Kaggle - DermNet](https://www.kaggle.com/datasets) & HAM10000
* 🛠 Built with: Python, Flask, TensorFlow, scikit-learn, HTML/CSS/JS

---

## ✍️ Authors

* Ishaan Taneja 

---

> *Empowering accessible and accurate skin disease diagnosis using AI.*

```

---

Let me know if you'd like a version with GitHub badges (e.g., license, technologies, stars) or a visual architecture diagram!
```
