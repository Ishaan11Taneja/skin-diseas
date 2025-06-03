# Skin Disease Detection AI

A comprehensive machine learning application for detecting and classifying various skin conditions using scikit-learn. The application provides a modern web interface for users to upload images and receive instant predictions with severity assessments.

## Features

- Image upload through drag-and-drop or file selection
- Real-time image analysis and classification
- Support for nine different skin conditions with severity levels:
  - High Severity:
    - Melanoma
    - Squamous cell carcinoma
  - Medium Severity:
    - Actinic keratosis
    - Atopic Dermatitis
    - Tinea Ringworm Candidiasis
  - Low Severity:
    - Benign keratosis
    - Dermatofibroma
    - Melanocytic nevus
    - Vascular lesion
- Probability scores for each condition
- Severity-based classification
- Modern and responsive web interface
- Detailed analysis reports

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd skin-disease-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
skin-disease-detection/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── train/
│   ├── Actinic keratosis/
│   ├── Atopic Dermatitis/
│   ├── Benign keratosis/
│   ├── Dermatofibroma/
│   ├── Melanocytic nevus/
│   ├── Melanoma/
│   ├── Squamous cell carcinoma/
│   ├── Tinea Ringworm Candidiasis/
│   └── Vascular lesion/
├── val/
│   └── [same structure as train/]
├── uploads/
├── app.py
├── train_model.py
├── download_kaggle_dataset.py
├── download_images.py
├── requirements.txt
├── skin_disease_model.joblib
├── class_names.txt
└── README.md
```

## Usage

1. Train the model:
```bash
python train_model.py
```
This will:
- Load and preprocess the training data
- Train the RandomForestClassifier model
- Save the trained model and class names
- Display performance metrics for each severity level

2. Start the web application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Upload an image of a skin condition and click "Analyze Image" to get the prediction.

## Model Details

The model uses scikit-learn's RandomForestClassifier with the following specifications:
- 200 decision trees
- Maximum depth of 20
- Balanced class weights
- Optimized for handling different severity levels

### Model Performance
- Overall accuracy: 64%
- Severity-based accuracy:
  - Low severity: 67.5%
  - Medium severity: 73.8%
  - High severity: 42.5%

## Data Collection

The project includes scripts for downloading and preparing the dataset:
- `download_kaggle_dataset.py`: Downloads the dataset from Kaggle
- `download_images.py`: Processes and organizes the downloaded images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle Skin Cancer Dataset]
- Built with Flask and scikit-learn
- Uses modern web technologies for the frontend 