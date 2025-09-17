import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def generate_mars_visualizations():
    """
    Generate visualization files for Mars water-ice detection
    """
    # Create model directory if it doesn't exist
    viz_dir = "data/models"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    print("Generating Mars training history visualization...")
    
    # Create sample training history visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sample data
    epochs = list(range(1, 26))
    train_acc = [0.70 + 0.012 * i for i in epochs]
    val_acc = [0.65 + 0.011 * i for i in epochs]
    train_loss = [1.0 - 0.032 * i for i in epochs]
    val_loss = [1.1 - 0.028 * i for i in epochs]
    train_precision = [0.72 + 0.011 * i for i in epochs]
    val_precision = [0.67 + 0.010 * i for i in epochs]
    train_recall = [0.69 + 0.013 * i for i in epochs]
    val_recall = [0.64 + 0.012 * i for i in epochs]
    
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
    
    # Plot precision
    ax3.plot(epochs, train_precision, 'bo-', label='Training Precision')
    ax3.plot(epochs, val_precision, 'ro-', label='Validation Precision')
    ax3.set_title('Model Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot recall
    ax4.plot(epochs, train_recall, 'bo-', label='Training Recall')
    ax4.plot(epochs, val_recall, 'ro-', label='Validation Recall')
    ax4.set_title('Model Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(viz_dir, 'mars_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mars training history visualization saved as '{output_path}'")
    
    print("Generating Mars confusion matrix visualization...")
    
    # Create confusion matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample confusion matrix data
    cm_data = [[88, 12], [16, 84]]
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
    ax.set_title('Mars Water-Ice Detection Confusion Matrix')
    
    output_path = os.path.join(viz_dir, 'mars_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mars confusion matrix visualization saved as '{output_path}'")

if __name__ == "__main__":
    try:
        generate_mars_visualizations()
        print("All Mars visualization files generated successfully!")
    except Exception as e:
        print(f"Error generating Mars visualization files: {e}")
        sys.exit(1)