import os
import requests
from PIL import Image
from io import BytesIO

# Sample image URLs for each condition
image_urls = {
    'acne': [
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/acne1.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/acne2.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/acne3.jpg'
    ],
    'eczema': [
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/eczema1.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/eczema2.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/eczema3.jpg'
    ],
    'melanoma': [
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/melanoma1.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/melanoma2.jpg',
        'https://raw.githubusercontent.com/ishaanmalhotra/skin-disease-detection/main/sample_images/melanoma3.jpg'
    ]
}

def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            print(f"Successfully downloaded: {save_path}")
        else:
            print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    # Create directories if they don't exist
    for condition in image_urls.keys():
        os.makedirs(os.path.join('data', 'train', condition), exist_ok=True)
    
    # Download images for each condition
    for condition, urls in image_urls.items():
        print(f"\nDownloading {condition} images...")
        for i, url in enumerate(urls, 1):
            save_path = os.path.join('data', 'train', condition, f'{condition}{i}.jpg')
            download_image(url, save_path)

if __name__ == "__main__":
    main() 