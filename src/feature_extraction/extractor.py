"""
LunarVision AI - Feature Extractor
=================================

This module handles feature extraction from satellite images for ice detection.
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(image_path):
    """
    Extract features from satellite images for ice detection
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        dict: Dictionary containing extracted features
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Edge detection using Canny edge detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    print("Extracted edges using Canny edge detector")
    
    # 2. Color histogram analysis
    # Calculate histograms for each color channel
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    print("Calculated color histograms for reflectance analysis")
    
    # 3. Texture analysis using gradients
    # Compute gradient magnitude and direction
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = np.arctan2(grad_y, grad_x)
    print("Computed gradient features for texture analysis")
    
    # 4. Feature vector preparation for ML model
    # Flatten and combine features
    edge_features = edges.flatten()
    hist_features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    texture_features = np.concatenate([gradient_magnitude.flatten(), gradient_direction.flatten()])
    
    # Combine all features
    feature_vector = np.concatenate([edge_features, hist_features, texture_features])
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()
    
    print(f"Feature extraction completed. Feature vector shape: {normalized_features.shape}")
    
    return {
        'edges': edges,
        'histograms': (hist_r, hist_g, hist_b),
        'gradients': (gradient_magnitude, gradient_direction),
        'feature_vector': normalized_features
    }

def main():
    """
    Main function to demonstrate feature extraction
    """
    print("LunarVision AI - Feature Extraction Module")
    print("=" * 45)
    
    # For demonstration, we'll use a dummy image
    # In practice, you would use an actual preprocessed satellite image
    import os
    dummy_image_path = "data/raw/dummy_satellite_image.jpg"
    
    if not os.path.exists(dummy_image_path):
        # Create a dummy image if it doesn't exist
        dummy_dir = "data/raw"
        if not os.path.exists(dummy_dir):
            os.makedirs(dummy_dir)
        
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_path, dummy_image)
    
    # Extract features
    try:
        features = extract_features(dummy_image_path)
        print("Feature extraction completed successfully")
        
        # Show feature info
        print(f"Edge map shape: {features['edges'].shape}")
        print(f"Feature vector length: {len(features['feature_vector'])}")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")

if __name__ == "__main__":
    main()