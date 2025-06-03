import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import joblib

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img.flatten()  # Flatten the image array

def get_severity_level(class_name):
    """Determine the severity level of a skin condition."""
    severity_map = {
        'Melanoma': 'high',
        'Squamous cell carcinoma': 'high',
        'Actinic keratosis': 'medium',
        'Atopic Dermatitis': 'medium',
        'Tinea Ringworm Candidiasis': 'medium',
        'Benign keratosis': 'low',
        'Dermatofibroma': 'low',
        'Melanocytic nevus': 'low',
        'Vascular lesion': 'low'
    }
    return severity_map.get(class_name, 'unknown')

def load_dataset(train_dir, val_dir):
    """Load the entire dataset from train and validation directories."""
    images = []
    labels = []
    severity_levels = []
    class_names = sorted(os.listdir(train_dir))
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load training data
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        print(f"Loading {class_name} images from {class_dir}")
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                    severity_levels.append(get_severity_level(class_name))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load validation data
    val_images = []
    val_labels = []
    val_severity_levels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        print(f"Loading {class_name} validation images from {class_dir}")
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = load_and_preprocess_image(img_path)
                    val_images.append(img)
                    val_labels.append(class_idx)
                    val_severity_levels.append(get_severity_level(class_name))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return (np.array(images), np.array(labels), np.array(severity_levels),
            np.array(val_images), np.array(val_labels), np.array(val_severity_levels),
            class_names)

def main():
    # Load the dataset
    print("Loading dataset...")
    X_train, y_train, severity_train, X_val, y_val, severity_val, class_names = load_dataset('train', 'val')
    
    print(f"\nDataset loaded successfully:")
    print(f"Training set size: {len(X_train)} images")
    print(f"Validation set size: {len(X_val)} images")
    
    # Print severity distribution
    print("\nSeverity Distribution in Training Set:")
    unique, counts = np.unique(severity_train, return_counts=True)
    for severity, count in zip(unique, counts):
        print(f"{severity}: {count} images")
    
    # Initialize and train the model with more trees and deeper depth
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better generalization
        max_depth=20,      # Deeper trees to capture complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    y_pred = model.predict(X_val)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Print severity-based accuracy
    print("\nSeverity-based Accuracy:")
    for severity in ['low', 'medium', 'high']:
        mask = severity_val == severity
        if np.any(mask):
            severity_acc = np.mean(y_pred[mask] == y_val[mask])
            print(f"{severity} severity accuracy: {severity_acc:.3f}")
    
    # Save the model
    print("\nSaving model...")
    joblib.dump(model, 'skin_disease_model.joblib')
    print("Model saved as 'skin_disease_model.joblib'")
    
    # Save class names and severity information
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            severity = get_severity_level(name)
            f.write(f"{name} ({severity})\n")
    print("Class names and severity levels saved to 'class_names.txt'")

if __name__ == "__main__":
    main() 



