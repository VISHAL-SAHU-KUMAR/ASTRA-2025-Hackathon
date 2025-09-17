"""
LunarVision AI - Image Preprocessor
==================================

This module handles preprocessing of satellite images for ice detection.
"""

import cv2
import numpy as np
import os

def preprocess_satellite_image(image_path, output_dir):
    """
    Preprocess satellite images for ice detection
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save processed images
    
    Returns:
        numpy.ndarray: Processed image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Original image shape: {image.shape}")
    
    # 1. Noise removal using Gaussian blur
    # Reduces high-frequency noise while preserving edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    print("Applied Gaussian blur for noise reduction")
    
    # 2. Brightness enhancement using histogram equalization
    # Improves contrast in the image
    # Convert to YUV color space for better luminance processing
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Equalize Y channel
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    print("Applied histogram equalization for brightness enhancement")
    
    # 3. Shadow correction using adaptive thresholding
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to correct shadows
    shadow_corrected = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    print("Applied adaptive thresholding for shadow correction")
    
    # Convert back to 3-channel for consistency
    shadow_corrected_3ch = cv2.cvtColor(shadow_corrected, cv2.COLOR_GRAY2BGR)
    
    # 4. Save output images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save intermediate results
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.jpg"), enhanced)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_shadow_corrected.jpg"), shadow_corrected_3ch)
    
    print(f"Saved processed images to {output_dir}")
    
    return shadow_corrected_3ch

def create_sample_data():
    """
    Create sample data for demonstration
    """
    # Create sample directory if it doesn't exist
    sample_dir = "data/raw"
    output_dir = "data/processed"
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For demonstration, we'll create a dummy image
    # In practice, you would use actual satellite imagery
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_path = os.path.join(sample_dir, "dummy_satellite_image.jpg")
    cv2.imwrite(dummy_path, dummy_image)
    
    return dummy_path, output_dir

def main():
    """
    Main function to demonstrate preprocessing
    """
    print("LunarVision AI - Image Preprocessing Module")
    print("=" * 45)
    
    # Create sample data
    dummy_path, output_dir = create_sample_data()
    
    # Process the image
    result = preprocess_satellite_image(dummy_path, output_dir)
    print("Image preprocessing completed successfully")
    
    # Show result info
    print(f"Processed image shape: {result.shape}")

if __name__ == "__main__":
    main()