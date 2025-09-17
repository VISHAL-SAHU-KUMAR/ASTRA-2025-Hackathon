# LunarVision AI Project Plan

## Prompt 1 - Project Overview

### 1. Problem Statement

**Why detecting ice on Moon and Mars is crucial:**
- **Resource Utilization**: Water ice can be converted to drinking water, oxygen for breathing, and hydrogen for fuel
- **Sustainable Human Presence**: Enables long-term habitation without Earth dependence
- **Mission Cost Reduction**: In-situ resource utilization (ISRU) reduces payload from Earth
- **Scientific Research**: Understanding water distribution provides insights into planetary formation and evolution

**Challenges in ice detection:**
- **Low Reflectance**: Ice may be mixed with regolith, reducing detectability
- **Surface Conditions**: Dust coverage, temperature variations, and radiation affect signatures
- **Resolution Limitations**: Satellite imagery may not provide sufficient detail
- **Ambiguous Spectral Signatures**: Other minerals may mimic ice signatures
- **Seasonal Variations**: Lighting conditions affect detection accuracy

### 2. Features List

1. **Image Enhancement Module**
   - Noise reduction and image sharpening
   - Contrast optimization for better feature visibility

2. **Texture Analysis Engine**
   - Surface roughness characterization
   - Pattern recognition for ice formations

3. **Reflectance Detection System**
   - Spectral signature analysis
   - Multi-wavelength comparison

4. **ML Classification Framework**
   - Supervised learning for labeled data
   - Binary classification (Ice/No Ice)

5. **Clustering Mechanism**
   - Unsupervised learning for unlabeled data
   - Region grouping based on similarity

6. **Visualization Dashboard**
   - Heatmap generation
   - 3D terrain representation

7. **Reporting System**
   - Automated report generation
   - Confidence scoring and interpretation

### 3. Project Structure

#### Module 1: Data Acquisition & Management
- **Tasks**: Dataset collection, organization, preprocessing
- **Tools**: NASA APIs, ESA portals, data simulation tools

#### Module 2: Image Preprocessing Pipeline
- **Tasks**: Noise removal, enhancement, normalization
- **Tools**: OpenCV, NumPy

#### Module 3: Feature Extraction Engine
- **Tasks**: Edge detection, texture analysis, spectral feature extraction
- **Tools**: OpenCV, Scikit-image

#### Module 4: ML Classification System
- **Tasks**: Model training, validation, testing
- **Tools**: TensorFlow/Keras, Scikit-learn

#### Module 5: Clustering Analysis Module
- **Tasks**: Unsupervised grouping, pattern identification
- **Tools**: Scikit-learn

#### Module 6: Visualization & Reporting
- **Tasks**: Heatmap generation, result presentation, report creation
- **Tools**: Matplotlib, Plotly, Flask/Django

### 4. Tools and Libraries

| Module | Tools/Libraries |
|--------|----------------|
| Data Acquisition | NASA API, ESA Portal, Planetary Data System |
| Image Processing | OpenCV, NumPy, SciPy |
| Feature Extraction | Scikit-image, Scikit-learn |
| ML Classification | TensorFlow, Keras, Scikit-learn |
| Clustering | Scikit-learn (K-means, DBSCAN) |
| Visualization | Matplotlib, Plotly, Seaborn |
| Web Interface | Flask/Django, HTML/CSS/JavaScript |
| Reporting | ReportLab, PDFKit |

## Prompt 2 - Dataset Collection

### Sources for Remote Sensing Data

1. **NASA Planetary Data System (PDS)**
   - URL: https://pds.nasa.gov/
   - Contains data from Lunar Reconnaissance Orbiter (LRO)
   - Diviner Lunar Radiometer data for temperature mapping

2. **NASA Earthdata Search**
   - URL: https://search.earthdata.nasa.gov/
   - Access to various satellite imagery datasets

3. **ESA Planetary Science Archive**
   - URL: https://archives.esac.esa.int/
   - Mars Express and Venus Express mission data

4. **USGS Astrogeology Science Center**
   - URL: https://astrogeology.usgs.gov/
   - High-resolution planetary imagery and maps

5. **Mars Orbital Laser Altimeter (MOLA)**
   - Topographic data for Mars surface

### Simulated Data Generation Process

If labeled datasets are unavailable:

1. **Synthetic Image Generation**
   ```python
   # Using procedural generation techniques
   # Create ice-like textures with known properties
   # Add realistic noise and artifacts
   ```

2. **Data Augmentation Techniques**
   - Rotation, scaling, and translation of existing images
   - Adding synthetic ice formations to real images
   - Adjusting lighting conditions and shadows

### Data Organization Structure

```
lunarvision_dataset/
├── raw/
│   ├── moon/
│   │   ├── lro_diviner/
│   │   ├── kaguya/
│   │   └── chang_e/
│   └── mars/
│       ├── hirise/
│       ├── crism/
│       └── ctx/
├── processed/
│   ├── moon/
│   └── mars/
├── labeled/
│   ├── ice/
│   └── no_ice/
├── metadata/
│   ├── moon_metadata.csv
│   └── mars_metadata.csv
└── simulated/
    ├── ice_samples/
    └── mixed_samples/
```

### File Naming Convention

```
[mission]_[instrument]_[date]_[coordinates]_[processing_level].[extension]
Example: lro_diviner_20230115_45.2N_32.1E_level2.tif
```

### Metadata Files

Create CSV files with columns:
- File path
- Mission name
- Instrument
- Acquisition date
- Coordinates
- Resolution
- Processing level
- Label (if available)

### Preprocessing Preparation

1. Convert all images to a consistent format (e.g., TIFF)
2. Normalize pixel values to [0,1] range
3. Apply radiometric calibration
4. Geometric correction and alignment
5. Create data manifests for tracking

## Prompt 3 - Image Preprocessing Code

```python
import cv2
import numpy as np
import os

def preprocess_satellite_image(image_path, output_dir):
    """
    Preprocess satellite images for ice detection
    """
    # Read the image
    image = cv2.imread(image_path)
    print(f"Original image shape: {image.shape}")
    
    # 1. Noise removal using Gaussian blur
    # Reduces high-frequency noise while preserving edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    print("Applied Gaussian blur for noise reduction")
    
    # Alternative: Median filter for salt-and-pepper noise
    # median_filtered = cv2.medianBlur(image, 5)
    # print("Applied median filter for noise reduction")
    
    # 2. Brightness enhancement using histogram equalization
    # Improves contrast in the image
    # Convert to YUV color space for better luminance processing
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Equalize Y channel
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    print("Applied histogram equalization for brightness enhancement")
    
    # Alternative: Contrast stretching
    # p2, p98 = np.percentile(blurred, (2, 98))
    # enhanced = np.clip((blurred - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    # print("Applied contrast stretching for brightness enhancement")
    
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

# Example usage with dummy image path
if __name__ == "__main__":
    # Create sample directory if it doesn't exist
    sample_dir = "sample_data"
    output_dir = "processed_data"
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For demonstration, we'll create a dummy image
    # In practice, you would use actual satellite imagery
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_path = os.path.join(sample_dir, "dummy_satellite_image.jpg")
    cv2.imwrite(dummy_path, dummy_image)
    
    # Process the image
    result = preprocess_satellite_image(dummy_path, output_dir)
    print("Image preprocessing completed successfully")
```

## Prompt 4 - Feature Extraction

```python
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def extract_features(image_path):
    """
    Extract features from satellite images for ice detection
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
    
    # Alternative: Sobel filter for edge detection
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # sobel_combined = np.hypot(sobelx, sobely)
    # print("Extracted edges using Sobel filter")
    
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

def visualize_features(image_path, features):
    """
    Visualize extracted features
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Edges
    axes[0, 1].imshow(features['edges'], cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    axes[0, 1].axis('off')
    
    # Color histograms
    axes[0, 2].plot(features['histograms'][0], color='red', alpha=0.7, label='Red')
    axes[0, 2].plot(features['histograms'][1], color='green', alpha=0.7, label='Green')
    axes[0, 2].plot(features['histograms'][2], color='blue', alpha=0.7, label='Blue')
    axes[0, 2].set_title('Color Histograms')
    axes[0, 2].legend()
    
    # Gradient magnitude
    axes[1, 0].imshow(features['gradients'][0], cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    # Gradient direction
    axes[1, 1].imshow(features['gradients'][1], cmap='hsv')
    axes[1, 1].set_title('Gradient Direction')
    axes[1, 1].axis('off')
    
    # Feature vector visualization
    axes[1, 2].plot(features['feature_vector'][:100])  # Show first 100 features
    axes[1, 2].set_title('Feature Vector (First 100 Elements)')
    
    plt.tight_layout()
    plt.savefig('feature_extraction_results.png')
    plt.show()
    
    print("Feature visualization saved as 'feature_extraction_results.png'")

# Example usage
if __name__ == "__main__":
    # Use the dummy image from preprocessing step
    dummy_path = "sample_data/dummy_satellite_image.jpg"
    
    # Extract features
    features = extract_features(dummy_path)
    
    # Visualize features
    visualize_features(dummy_path, features)
    
    print("Feature extraction and visualization completed")
```

## Prompt 5 - Model Architecture (Classification)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

def create_cnn_model(input_shape=(256, 256, 3), num_classes=2):
    """
    Create a CNN model for ice detection in satellite images
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Binary classification
    ])
    
    print("CNN model architecture created")
    model.summary()
    
    return model

def prepare_dummy_dataset(num_samples=1000, img_height=256, img_width=256):
    """
    Create dummy dataset for demonstration
    In practice, you would load actual satellite images
    """
    # Generate dummy images
    X = np.random.rand(num_samples, img_height, img_width, 3).astype(np.float32)
    
    # Generate dummy labels (0: No Ice, 1: Ice)
    y = np.random.randint(0, 2, num_samples)
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=2)
    
    print(f"Dummy dataset created: {X.shape[0]} samples")
    print(f"Image shape: {X.shape[1:]}")
    print(f"Label distribution: No Ice={np.sum(y==0)}, Ice={np.sum(y==1)}")
    
    return X, y_categorical

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the CNN model
    """
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled with:")
    print("- Optimizer: Adam")
    print("- Loss function: Categorical Crossentropy")
    print("- Metrics: Accuracy")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return history

# Example usage
if __name__ == "__main__":
    # Create the model
    model = create_cnn_model(input_shape=(256, 256, 3), num_classes=2)
    
    # Prepare dummy dataset
    X, y = prepare_dummy_dataset(num_samples=1000)
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train the model
    print("\nStarting model training...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=5)
    
    # Save the model
    model.save('lunarvision_ice_classifier.h5')
    print("\nModel saved as 'lunarvision_ice_classifier.h5'")
    
    print("CNN model training completed")
```

## Prompt 6 - Clustering Implementation (Unsupervised Learning)

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2

def extract_features_for_clustering(image_path):
    """
    Extract features suitable for clustering
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to different color spaces
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract statistical features
    features = []
    
    # RGB mean and std
    for i in range(3):
        features.extend([np.mean(image_rgb[:,:,i]), np.std(image_rgb[:,:,i])])
    
    # HSV mean and std
    for i in range(3):
        features.extend([np.mean(image_hsv[:,:,i]), np.std(image_hsv[:,:,i])])
    
    # LAB mean and std
    for i in range(3):
        features.extend([np.mean(image_lab[:,:,i]), np.std(image_lab[:,:,i])])
    
    # Texture features using Local Binary Pattern (simplified)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.extend([np.mean(gray), np.std(gray)])
    
    return np.array(features)

def create_sample_data(num_samples=200):
    """
    Create sample feature vectors for clustering demonstration
    """
    # In practice, you would extract features from actual satellite images
    # Here we create synthetic feature vectors
    np.random.seed(42)
    
    # Create two distinct clusters of feature vectors
    cluster1 = np.random.normal(100, 20, (num_samples//2, 10))  # Ice-like features
    cluster2 = np.random.normal(50, 15, (num_samples//2, 10))   # Non-ice features
    
    # Combine clusters
    feature_vectors = np.vstack([cluster1, cluster2])
    
    # Add some noise
    feature_vectors += np.random.normal(0, 5, feature_vectors.shape)
    
    return feature_vectors

def perform_kmeans_clustering(feature_vectors, n_clusters=2):
    """
    Perform K-means clustering on feature vectors
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    print(f"K-means clustering completed with {n_clusters} clusters")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    return cluster_labels, kmeans, scaler

def perform_dbscan_clustering(feature_vectors, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on feature vectors
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(scaled_features)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN clustering completed with {n_clusters} clusters and {n_noise} noise points")
    
    return cluster_labels, dbscan, scaler

def visualize_clusters(feature_vectors, cluster_labels, method_name):
    """
    Visualize clustering results using PCA for dimensionality reduction
    """
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(feature_vectors)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'{method_name} Clustering Results (PCA Visualization)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{method_name.lower()}_clustering_results.png')
    plt.show()
    
    print(f"Clustering visualization saved as '{method_name.lower()}_clustering_results.png'")

def analyze_clusters(feature_vectors, cluster_labels):
    """
    Analyze clusters to identify potential ice regions
    """
    unique_clusters = np.unique(cluster_labels)
    
    print("\nCluster Analysis:")
    print("-" * 40)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_points = feature_vectors[cluster_labels == cluster_id]
            print(f"Noise points: {len(cluster_points)} samples")
        else:
            cluster_points = feature_vectors[cluster_labels == cluster_id]
            mean_features = np.mean(cluster_points, axis=0)
            print(f"Cluster {cluster_id}: {len(cluster_points)} samples")
            print(f"  Mean feature values: {mean_features[:5]}...")  # Show first 5 features
            
            # Heuristic: Cluster with higher mean values might indicate ice
            # This would need to be validated with domain knowledge
            if np.mean(mean_features) > np.mean(feature_vectors):
                print(f"  Potential ice cluster (higher reflectance values)")
            else:
                print(f"  Likely non-ice cluster")

# Example usage
if __name__ == "__main__":
    # Create sample data
    print("Creating sample feature vectors...")
    feature_vectors = create_sample_data(num_samples=200)
    print(f"Feature vectors shape: {feature_vectors.shape}")
    
    # K-means clustering
    print("\n" + "="*50)
    print("PERFORMING K-MEANS CLUSTERING")
    print("="*50)
    
    kmeans_labels, kmeans_model, kmeans_scaler = perform_kmeans_clustering(feature_vectors, n_clusters=2)
    visualize_clusters(feature_vectors, kmeans_labels, "K-Means")
    analyze_clusters(feature_vectors, kmeans_labels)
    
    # DBSCAN clustering
    print("\n" + "="*50)
    print("PERFORMING DBSCAN CLUSTERING")
    print("="*50)
    
    dbscan_labels, dbscan_model, dbscan_scaler = perform_dbscan_clustering(feature_vectors, eps=2.0, min_samples=10)
    visualize_clusters(feature_vectors, dbscan_labels, "DBSCAN")
    analyze_clusters(feature_vectors, dbscan_labels)
    
    print("\nClustering analysis completed")
```

## Prompt 7 - Visualization & Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

def generate_heatmap(image_path, confidence_scores, output_path="heatmap_result.png"):
    """
    Generate a heatmap overlay on the satellite image
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
    plt.show()
    
    print(f"Heatmap visualization saved as '{output_path}'")

def plot_confidence_distribution(confidence_scores, output_path="confidence_distribution.png"):
    """
    Plot the distribution of confidence scores
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
    plt.show()
    
    print(f"Confidence distribution plot saved as '{output_path}'")

def create_3d_visualization(image_path, elevation_data, output_path="3d_terrain.png"):
    """
    Create a basic 3D terrain visualization
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("3D visualization requires mpl_toolkits.mplot3d")
        return
    
    # Read the image
    image = cv2.imread(image_path)
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
    plt.show()
    
    print(f"3D visualization saved as '{output_path}'")

# Example usage
if __name__ == "__main__":
    # Create dummy data for demonstration
    dummy_image_path = "sample_data/dummy_satellite_image.jpg"
    
    # Create sample directory and dummy image if needed
    sample_dir = "sample_data"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
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
    generate_heatmap(dummy_image_path, dummy_confidence)
    
    # Plot confidence distribution
    print("Plotting confidence distribution...")
    plot_confidence_distribution(dummy_confidence)
    
    # Create 3D visualization
    print("Creating 3D terrain visualization...")
    create_3d_visualization(dummy_image_path, None)
    
    print("Visualization examples completed")
```

## Prompt 8 - User Interface / Report Generation

```python
from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """
    Main page with image upload functionality
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and process it
    """
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image (dummy processing for demonstration)
        processed_image_path = process_image(filepath)
        
        # Generate results (dummy results for demonstration)
        results = {
            'ice_probability': 0.78,
            'confidence_score': 0.85,
            'region_count': 3,
            'coordinates': [(120, 240), (300, 400), (450, 180)]
        }
        
        # Save results
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{file.filename}")
        # In practice, you would save actual processed images
        
        return render_template('results.html', 
                             filename=file.filename,
                             results=results,
                             processed_image=processed_image_path)

def process_image(image_path):
    """
    Process the uploaded image (dummy implementation)
    In practice, this would call your preprocessing, feature extraction,
    and ML model functions
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Apply some dummy processing
    processed = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Save processed image
    filename = os.path.basename(image_path)
    processed_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    cv2.imwrite(processed_path, processed)
    
    return processed_path

@app.route('/generate_report/<filename>')
def generate_report(filename):
    """
    Generate PDF report for the analysis
    """
    # Create PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "LunarVision AI - Ice Detection Report")
    
    # Add basic information
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Image: {filename}")
    c.drawString(50, height - 120, "Analysis Date: 2025-09-17")
    c.drawString(50, height - 140, "Detected Ice Probability: 78%")
    c.drawString(50, height - 160, "Confidence Score: 85%")
    c.drawString(50, height - 180, "Potential Ice Regions: 3")
    
    # Add dummy chart (in practice, you would add actual visualizations)
    c.drawString(50, height - 220, "Confidence Distribution:")
    c.rect(50, height - 300, 200, 50)  # Dummy bar chart
    c.drawString(60, height - 280, "High Confidence: 60%")
    c.drawString(60, height - 260, "Medium Confidence: 25%")
    c.drawString(60, height - 240, "Low Confidence: 15%")
    
    # Add summary
    c.drawString(50, height - 350, "Summary:")
    c.drawString(70, height - 370, "• Ice detected in 3 distinct regions")
    c.drawString(70, height - 390, "• Highest confidence in central region")
    c.drawString(70, height - 410, "• Recommended for further investigation")
    
    # Save PDF
    c.save()
    
    # Return PDF
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"ice_detection_report_{filename}.pdf", mimetype='application/pdf')

# HTML templates would be stored in a 'templates' folder
# For demonstration, here are the template contents:

def create_templates():
    """
    Create HTML templates for the web interface
    """
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # Main page template
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>LunarVision AI - Ice Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f0f8ff; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #4682b4; text-align: center; }
        .upload-area { border: 2px dashed #4682b4; padding: 30px; text-align: center; margin: 20px 0; border-radius: 5px; }
        input[type="file"] { margin: 20px 0; }
        input[type="submit"] { background-color: #4682b4; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #5a9bd4; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LunarVision AI - Lunar and Martian Ice Detection</h1>
        <p>Upload a satellite image to detect potential water ice on Moon or Mars surfaces.</p>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="upload-area">
                <p>Select satellite image file:</p>
                <input type="file" name="file" accept="image/*" required>
            </div>
            <input type="submit" value="Analyze for Ice">
        </form>
    </div>
</body>
</html>
    """
    
    # Results page template
    results_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results - LunarVision AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f0f8ff; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #4682b4; text-align: center; }
        .result-box { background-color: #e6f3ff; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .result-item { margin: 10px 0; }
        .btn { background-color: #4682b4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }
        .btn:hover { background-color: #5a9bd4; }
        img { max-width: 100%; height: auto; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ice Detection Results</h1>
        
        <div class="result-box">
            <h2>Analysis Summary</h2>
            <div class="result-item"><strong>Image:</strong> {{ filename }}</div>
            <div class="result-item"><strong>Ice Probability:</strong> {{ results.ice_probability * 100 }}%</div>
            <div class="result-item"><strong>Confidence Score:</strong> {{ results.confidence_score * 100 }}%</div>
            <div class="result-item"><strong>Potential Ice Regions:</strong> {{ results.region_count }}</div>
        </div>
        
        <div class="result-box">
            <h2>Processed Image</h2>
            <!-- In practice, you would display the actual processed image -->
            <p>Processed image would be displayed here</p>
        </div>
        
        <div style="text-align: center;">
            <a href="/generate_report/{{ filename }}" class="btn">Download PDF Report</a>
            <a href="/" class="btn">Analyze Another Image</a>
        </div>
    </div>
</body>
</html>
    """
    
    # Write templates to files
    with open(os.path.join(templates_dir, "index.html"), "w") as f:
        f.write(index_html)
    
    with open(os.path.join(templates_dir, "results.html"), "w") as f:
        f.write(results_html)
    
    print("Web templates created successfully")

# Example usage
if __name__ == "__main__":
    # Create templates
    create_templates()
    
    print("Flask web application structure created")
    print("To run the application, execute: python app.py")
    print("Then visit http://localhost:5000 in your browser")
    
    # To actually run the Flask app, you would uncomment these lines:
    # app.run(debug=True)
```

## Prompt 9 - Future Scope

### 1. Environmental Data Integration

**Incorporating Environmental Parameters:**

1. **Temperature Data Integration**
   - Integrate thermal imaging data from instruments like LRO's Diviner
   - Correlate surface temperature with ice stability models
   - Implement temperature-based filtering for ice probability

2. **Lighting Condition Modeling**
   - Account for solar incidence angles affecting reflectance
   - Model seasonal variations in illumination
   - Adjust detection algorithms based on local time of day

3. **Atmospheric/Humidity Factors (Mars-specific)**
   - Incorporate atmospheric pressure data
   - Model water vapor cycles in the Martian atmosphere
   - Correlate with subsurface ice stability

**Implementation Approach:**
```python
# Example of environmental data integration
def integrate_environmental_data(base_features, temp_data, illumination_angle):
    """
    Enhance feature vectors with environmental context
    """
    # Normalize environmental parameters
    normalized_temp = (temp_data - temp_data.min()) / (temp_data.max() - temp_data.min())
    normalized_illumination = illumination_angle / 90.0  # Assuming max 90 degrees
    
    # Combine with base features
    enhanced_features = np.column_stack([
        base_features,
        normalized_temp.flatten(),
        normalized_illumination.flatten()
    ])
    
    return enhanced_features
```

### 2. Real-time Spacecraft Integration

**Integration with Space Missions:**

1. **Rover-based Real-time Analysis**
   - Deploy lightweight models on rover computing systems
   - Implement edge computing for immediate decision-making
   - Enable autonomous navigation toward high-probability ice regions

2. **Orbiter Data Streaming**
   - Process continuous data streams from orbiters
   - Implement change detection for monitoring seasonal variations
   - Enable predictive modeling for future observations

3. **Communication Protocols**
   - Develop protocols for Earth-spacecraft data exchange
   - Implement error correction for deep-space communications
   - Optimize data compression for bandwidth limitations

**Technical Implementation:**
```python
# Example of spacecraft integration framework
class SpacecraftIntegration:
    def __init__(self):
        self.model = self.load_optimized_model()
    
    def load_optimized_model(self):
        """
        Load a lightweight model suitable for spacecraft computing
        """
        # Use TensorFlow Lite or similar for edge deployment
        # model = tf.lite.Interpreter(model_path="optimized_model.tflite")
        # model.allocate_tensors()
        # return model
        pass
    
    def process_realtime_data(self, image_data, telemetry):
        """
        Process incoming data from spacecraft instruments
        """
        # Preprocess image
        processed_image = self.preprocess(image_data)
        
        # Extract features with environmental context
        features = self.extract_features(processed_image, telemetry)
        
        # Run inference
        prediction = self.model.predict(features)
        
        # Format results for transmission
        results = self.format_results(prediction, telemetry)
        
        return results
```

### 3. Extended Mineral and Gas Detection

**Beyond Ice Detection:**

1. **Multi-Mineral Classification**
   - Extend CNN architecture for multi-class mineral identification
   - Incorporate spectral libraries for known minerals
   - Implement uncertainty quantification for rare minerals

2. **Gas Detection Capabilities**
   - Analyze atmospheric composition from spectral data
   - Detect trace gases that may indicate subsurface activity
   - Correlate gas concentrations with geological features

3. **Organic Compound Identification**
   - Search for biosignatures in hyperspectral data
   - Implement specialized algorithms for complex organic molecules
   - Develop contamination detection to ensure data integrity

**Model Architecture Extension:**
```python
# Example of extended classification model
def create_extended_model(input_shape, num_classes):
    """
    Create a model for multi-class mineral/gas detection
    """
    model = tf.keras.Sequential([
        # Feature extraction layers (shared with ice detection)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Multi-head output for different materials
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Multiple output heads
        tf.keras.layers.Dense(num_classes['ice'], activation='softmax', name='ice_output'),
        tf.keras.layers.Dense(num_classes['minerals'], activation='softmax', name='mineral_output'),
        tf.keras.layers.Dense(num_classes['gases'], activation='sigmoid', name='gas_output')
    ])
    
    return model
```

### 4. Open-Source Collaboration Platform

**Building a Collaborative Ecosystem:**

1. **Version Control and Collaboration Tools**
   - Git-based workflow for model development
   - GitHub/GitLab integration for community contributions
   - Continuous integration for automated testing

2. **Data Sharing Infrastructure**
   - Implement federated learning for distributed data
   - Create standardized data formats for interoperability
   - Develop APIs for third-party tool integration

3. **Community Engagement Roadmap**
   - Documentation portal with tutorials and examples
   - Regular hackathons and coding challenges
   - Academic collaboration programs with research institutions

**Platform Architecture:**
```yaml
# Example platform structure
lunarvision_platform:
  core_engine:
    preprocessing_module: "OpenCV-based image processing"
    feature_extraction: "Scikit-learn and custom algorithms"
    ml_models: "TensorFlow/Keras neural networks"
  
  collaboration_tools:
    version_control: "Git with GitLab integration"
    ci_cd: "Automated testing and deployment"
    documentation: "Sphinx-based documentation system"
  
  api_services:
    rest_api: "Flask-based RESTful services"
    data_exchange: "Standardized data formats (GeoTIFF, NetCDF)"
    model_serving: "TensorFlow Serving for model deployment"
  
  community_features:
    forums: "Discussion boards for researchers"
    tutorials: "Interactive Jupyter notebooks"
    challenges: "Competitions for model improvement"
```

**Implementation Timeline:**

1. **Phase 1 (Months 1-6)**: Core platform development
   - Establish Git repository with basic project structure
   - Implement core ML pipeline components
   - Create initial documentation

2. **Phase 2 (Months 7-12)**: Community features
   - Launch public repository
   - Implement contribution guidelines
   - Host first community hackathon

3. **Phase 3 (Months 13-18)**: Advanced capabilities
   - Integrate environmental data processing
   - Implement multi-material detection
   - Establish partnerships with space agencies

4. **Phase 4 (Months 19-24)**: Production deployment
   - Deploy platform for active mission support
   - Implement real-time processing capabilities
   - Scale infrastructure for global collaboration

This roadmap provides a comprehensive foundation for the LunarVision AI project, enabling both immediate implementation and long-term expansion into a collaborative platform for space exploration.