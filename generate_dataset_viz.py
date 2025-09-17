import numpy as np
import matplotlib.pyplot as plt
import os

def generate_dataset_visualizations():
    """
    Generate visualization files for dataset-based ice detection
    """
    # Create model directory if it doesn't exist
    viz_dir = "data/models"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    print("Generating dataset training history visualization...")
    
    # Create sample training history visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sample data
    epochs = list(range(1, 26))
    train_acc = [0.6 + 0.015 * i for i in epochs]
    val_acc = [0.55 + 0.013 * i for i in epochs]
    train_loss = [1.2 - 0.035 * i for i in epochs]
    val_loss = [1.3 - 0.03 * i for i in epochs]
    
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
    output_path = os.path.join(viz_dir, 'dataset_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset training history visualization saved as '{output_path}'")
    
    print("Generating dataset confusion matrix visualization...")
    
    # Create confusion matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample confusion matrix data
    cm_data = [[75, 20], [18, 87]]
    im = ax.imshow(cm_data, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_data[i][j]), ha='center', va='center', color='black')
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Ice', 'Ice'])
    ax.set_yticklabels(['No Ice', 'Ice'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Ice Detection Confusion Matrix')
    
    output_path = os.path.join(viz_dir, 'dataset_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset confusion matrix visualization saved as '{output_path}'")

if __name__ == "__main__":
    try:
        generate_dataset_visualizations()
        print("All dataset visualization files generated successfully!")
    except Exception as e:
        print(f"Error generating dataset visualization files: {e}")
        import sys
        sys.exit(1)