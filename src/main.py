"""
LunarVision AI - Main Application
=================================

This is the main entry point for the LunarVision AI system.
It demonstrates how to use the different components of the system.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def main():
    """
    Main function to run the LunarVision AI system
    """
    print("LunarVision AI - Water Ice Detection System")
    print("=" * 45)
    print("Initializing system...")
    
    # Check if required directories exist
    required_dirs = ['data', 'src', 'notebooks']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Warning: Directory '{dir_name}' not found")
    
    print("\nSystem components:")
    print("1. Data preprocessing module")
    print("2. Feature extraction engine")
    print("3. Machine learning models")
    print("4. Visualization tools")
    print("5. Web interface")
    
    print("\nTo run specific components, use:")
    print("  python src/preprocessing/preprocessor.py")
    print("  python src/feature_extraction/extractor.py")
    print("  python src/models/trainer.py")
    print("  python src/visualization/visualizer.py")
    print("  python src/web_interface/app.py")
    
    print("\nFor more information, check the documentation in the docs/ directory")
    print("or refer to the project plan in LunarVision_AI_Project_Plan.md")

if __name__ == "__main__":
    main()