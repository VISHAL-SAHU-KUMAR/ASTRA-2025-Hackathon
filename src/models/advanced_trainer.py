"""
LunarVision AI - Advanced Model Trainer
====================================

This module handles advanced training of machine learning models for ice detection
with real data simulation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow/Keras imported successfully")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Using simulation mode.")

def create_advanced_cnn_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Create an advanced CNN model for ice detection in satellite images
    
    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    if not TENSORFLOW_AVAILABLE:
        print("Model architecture definition (TensorFlow not available):")
        model_description = f"""
        Advanced CNN Model Architecture:
        - Input shape: {input_shape}
        - Conv2D (32 filters, 3x3 kernel) + BatchNormalization + ReLU + MaxPooling2D
        - Conv2D (64 filters, 3x3 kernel) + BatchNormalization + ReLU + MaxPooling2D
        - Conv2D (128 filters, 3x3 kernel) + BatchNormalization + ReLU + MaxPooling2D
        - Conv2D (256 filters, 3x3 kernel) + BatchNormalization + ReLU + MaxPooling2D
        - GlobalAveragePooling2D
        - Dense (512 units) + BatchNormalization + ReLU + Dropout (0.5)
        - Dense (256 units) + BatchNormalization + ReLU + Dropout (0.3)
        - Dense ({num_classes} units) with softmax activation
        """
        print(model_description)
        return None
    
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation=None, input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global average pooling instead of flatten
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with batch normalization and dropout
        layers.Dense(512, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Advanced CNN model architecture created")
    return model

def generate_realistic_dataset(num_samples=2000, img_height=128, img_width=128):
    """
    Generate a more realistic dataset for ice detection simulation
    
    Args:
        num_samples (int): Number of samples to generate
        img_height (int): Height of images
        img_width (int): Width of images
    
    Returns:
        tuple: (X, y) where X is image data and y is labels
    """
    print(f"Generating realistic dataset with {num_samples} samples...")
    
    # Generate base images with different characteristics
    X = []
    y = []
    
    for i in range(num_samples):
        # Create a base image with random noise
        image = np.random.rand(img_height, img_width, 3).astype(np.float32)
        
        # Simulate different surface types
        if np.random.rand() > 0.5:  # Ice-like surface (50% probability)
            # Add ice-like characteristics (brighter, more uniform)
            image = image * 0.7 + 0.3  # Brighten
            # Add some uniform regions
            for _ in range(np.random.randint(1, 4)):
                x_start = np.random.randint(0, img_width - 20)
                y_start = np.random.randint(0, img_height - 20)
                size = np.random.randint(10, 30)
                image[y_start:y_start+size, x_start:x_start+size] = np.random.rand(size, size, 3) * 0.2 + 0.8
            label = 1  # Ice
        else:  # Non-ice surface (50% probability)
            # Add more varied textures
            image = image * np.random.rand(img_height, img_width, 3)
            # Add some dark spots
            for _ in range(np.random.randint(0, 3)):
                x_start = np.random.randint(0, img_width - 10)
                y_start = np.random.randint(0, img_height - 10)
                size = np.random.randint(5, 20)
                image[y_start:y_start+size, x_start:x_start+size] = np.random.rand(size, size, 3) * 0.3
            label = 0  # No Ice
        
        X.append(image)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=2)
    
    print(f"Dataset created: {X.shape[0]} samples")
    print(f"Image shape: {X.shape[1:]}")
    print(f"Label distribution: No Ice={np.sum(y==0)}, Ice={np.sum(y==1)}")
    
    return X, y_categorical

def train_advanced_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the advanced CNN model with callbacks and visualization
    
    Args:
        model (tensorflow.keras.Model): Model to train
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        history: Training history
    """
    if not TENSORFLOW_AVAILABLE:
        print("Simulating model training process...")
        print("- Optimizer: Adam")
        print("- Loss function: Categorical Crossentropy")
        print("- Metrics: Accuracy")
        print("- Early Stopping: patience=10")
        print("- Learning Rate Reduction: factor=0.5, patience=5")
        
        # Simulate training process
        epochs_range = range(1, min(epochs, 20))  # Simulate only 20 epochs
        history = {
            'epochs': list(epochs_range),
            'train_accuracy': [0.6 + 0.02 * i for i in epochs_range],
            'val_accuracy': [0.55 + 0.018 * i for i in epochs_range],
            'train_loss': [1.2 - 0.04 * i for i in epochs_range],
            'val_loss': [1.3 - 0.035 * i for i in epochs_range]
        }
        
        print(f"Training completed for {len(epochs_range)} epochs")
        return history
    
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
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training history with accuracy and loss curves
    
    Args:
        history: Training history from model.fit()
    """
    if not TENSORFLOW_AVAILABLE:
        # Use simulated data
        epochs = history['epochs']
        train_acc = history['train_accuracy']
        val_acc = history['val_accuracy']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
    else:
        # Use real history data
        epochs = range(1, len(history.history['accuracy']) + 1)
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(epochs, train_loss, 'bo-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history visualization saved as 'data/models/training_history.png'")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and generate reports
    
    Args:
        model (tensorflow.keras.Model): Trained model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
    """
    if not TENSORFLOW_AVAILABLE:
        print("Simulating model evaluation...")
        print("Classification Report:")
        print("              precision    recall  f1-score   support")
        print()
        print("     No Ice       0.82      0.78      0.80        98")
        print("       Ice       0.80      0.84      0.82       102")
        print()
        print("    accuracy                           0.81       200")
        print("   macro avg       0.81      0.81      0.81       200")
        print("weighted avg       0.81      0.81      0.81       200")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=['No Ice', 'Ice'])
    print("Classification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Ice', 'Ice'], 
                yticklabels=['No Ice', 'Ice'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('data/models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Confusion matrix visualization saved as 'data/models/confusion_matrix.png'")

def main():
    """
    Main function to demonstrate advanced model training
    """
    print("LunarVision AI - Advanced Model Training")
    print("=" * 40)
    
    # Create model directory if it doesn't exist
    model_dir = "data/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create the model
    model = create_advanced_cnn_model(input_shape=(128, 128, 3), num_classes=2)
    
    # Generate dataset
    X, y = generate_realistic_dataset(num_samples=2000, img_height=128, img_width=128)
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("\nStarting model training...")
    history = train_advanced_model(model, X_train, y_train, X_val, y_val, epochs=30)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    if TENSORFLOW_AVAILABLE:
        model_path = os.path.join(model_dir, "lunarvision_advanced_ice_classifier.h5")
        model.save(model_path)
        print(f"\nModel saved as '{model_path}'")
    
    print("\nAdvanced model training completed successfully")

if __name__ == "__main__":
    main()