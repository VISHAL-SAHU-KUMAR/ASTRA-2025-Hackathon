"""
LunarVision AI - Visualizer
==========================

This module handles visualization of results from the ice detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def generate_heatmap(image_path, confidence_scores, output_path="heatmap_result.png"):
    """
    Generate a heatmap overlay on the satellite image
    
    Args:
        image_path (str): Path to the original image
        confidence_scores (numpy.ndarray): Confidence scores for ice detection
        output_path (str): Path to save the heatmap image
    """
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create heatmap based on confidence scores
    # For demonstration, we'll create a dummy confidence map
    height, width = original_image_rgb.shape[:2]
    
    # Resize confidence scores to match image dimensions
    if len(confidence_scores.shape) == 1:
        # If 1D array, reshape to 2D
        side = int(np.sqrt(len(confidence_scores)))
        confidence_map = confidence_scores[:side*side].reshape(side, side)
        confidence_map = cv2.resize(confidence_map, (width, height))
    else:
        confidence_map = cv2.resize(confidence_scores, (width, height))
    
    # Create heatmap overlay
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Satellite Image')
    plt.axis('off')
    
    # Confidence heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(confidence_map, cmap='hot', interpolation='nearest')
    plt.title('Ice Detection Confidence')
    plt.colorbar()
    plt.axis('off')
    
    # Overlay heatmap on original image
    plt.subplot(1, 3, 3)
    plt.imshow(original_image_rgb)
    plt.imshow(confidence_map, cmap='hot', alpha=0.6)  # Overlay with transparency
    plt.title('Heatmap Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Heatmap visualization saved as '{output_path}'")

def plot_confidence_distribution(confidence_scores, output_path="confidence_distribution.png"):
    """
    Plot the distribution of confidence scores
    
    Args:
        confidence_scores (numpy.ndarray): Confidence scores for ice detection
        output_path (str): Path to save the distribution plot
    """
    plt.figure(figsize=(10, 6))
    
    # Histogram of confidence scores
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ice Detection Confidence Scores')
    plt.grid(True, alpha=0.3)
    
    # Box plot of confidence scores
    plt.subplot(1, 2, 2)
    plt.boxplot(confidence_scores.flatten())
    plt.ylabel('Confidence Score')
    plt.title('Confidence Score Distribution (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Confidence distribution plot saved as '{output_path}'")

def create_3d_visualization(image_path, elevation_data=None, output_path="3d_terrain.png"):
    """
    Create a basic 3D terrain visualization
    
    Args:
        image_path (str): Path to the satellite image
        elevation_data (numpy.ndarray): Elevation data (optional)
        output_path (str): Path to save the 3D visualization
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        mpl_toolkits_available = True
    except ImportError:
        mpl_toolkits_available = False
        print("Warning: mpl_toolkits not available. 3D visualization will be limited.")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        # Create a dummy image if the file doesn't exist
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create coordinate grids
    height, width = image_rgb.shape[:2]
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y)
    
    # For demonstration, create synthetic elevation data if not provided
    if elevation_data is None:
        # Create synthetic elevation with some "ice-like" peaks
        elevation_data = np.sin(X/50) * np.cos(Y/50) * 50
        # Add some random peaks
        for _ in range(5):
            cx, cy = np.random.randint(0, width), np.random.randint(0, height)
            elevation_data += 100 * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*50**2))
    
    if mpl_toolkits_available:
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Downsample for better performance
        step = max(1, width // 100)
        X_ds, Y_ds = X[::step, ::step], Y[::step, ::step]
        Z_ds = elevation_data[::step, ::step]
        
        # Plot the surface
        surf = ax.plot_surface(X_ds, Y_ds, Z_ds, cmap='terrain', alpha=0.8)
        
        # Add color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain Visualization with Potential Ice Regions')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"3D visualization saved as '{output_path}'")
    else:
        # Create a 2D representation if 3D is not available
        plt.figure(figsize=(10, 8))
        plt.imshow(elevation_data, cmap='terrain')
        plt.colorbar(label='Elevation')
        plt.title('2D Terrain Visualization (3D not available)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"2D visualization saved as '{output_path}' (3D not available)")

def main():
    """
    Main function to demonstrate visualization capabilities
    """
    print("LunarVision AI - Visualization Module")
    print("=" * 35)
    
    # Create sample data for demonstration
    sample_dir = "data/raw"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    dummy_image_path = os.path.join(sample_dir, "dummy_satellite_image.jpg")
    
    if not os.path.exists(dummy_image_path):
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    # Generate dummy confidence scores
    dummy_confidence = np.random.rand(100, 100)  # 100x100 grid of confidence scores
    # Add some "ice-like" high confidence regions
    dummy_confidence[30:70, 30:70] += 0.5  # Central region
    dummy_confidence = np.clip(dummy_confidence, 0, 1)  # Ensure values are in [0,1]
    
    # Generate heatmap
    print("Generating heatmap visualization...")
    generate_heatmap(dummy_image_path, dummy_confidence, "data/processed/heatmap_result.png")
    
    # Plot confidence distribution
    print("Plotting confidence distribution...")
    plot_confidence_distribution(dummy_confidence, "data/processed/confidence_distribution.png")
    
    # Create 3D visualization
    print("Creating terrain visualization...")
    create_3d_visualization(dummy_image_path, None, "data/processed/3d_terrain.png")
    
    print("Visualization examples completed")

if __name__ == "__main__":
    main()