import os
import kaggle
import zipfile
import shutil

def download_dataset():
    """Download the skin disease dataset from Kaggle."""
    print("Downloading dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    kaggle.api.dataset_download_files(
        'shubhamgoel27/dermnet',
        path='data',
        unzip=True
    )
    
    # Create train and validation directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)
    
    # Move images to appropriate directories
    source_dir = 'data/dermnet'
    train_dir = 'data/train'
    val_dir = 'data/validation'
    
    # Process each skin condition
    for condition in os.listdir(source_dir):
        condition_dir = os.path.join(source_dir, condition)
        if os.path.isdir(condition_dir):
            # Create condition directories in train and validation
            os.makedirs(os.path.join(train_dir, condition), exist_ok=True)
            os.makedirs(os.path.join(val_dir, condition), exist_ok=True)
            
            # Get all images for this condition
            images = [f for f in os.listdir(condition_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Split into train and validation (80-20 split)
            train_size = int(0.8 * len(images))
            train_images = images[:train_size]
            val_images = images[train_size:]
            
            # Move images to appropriate directories
            for img in train_images:
                src = os.path.join(condition_dir, img)
                dst = os.path.join(train_dir, condition, img)
                shutil.copy2(src, dst)
            
            for img in val_images:
                src = os.path.join(condition_dir, img)
                dst = os.path.join(val_dir, condition, img)
                shutil.copy2(src, dst)
    
    # Clean up
    shutil.rmtree(source_dir)
    print("Dataset preparation completed!")

if __name__ == "__main__":
    download_dataset() 