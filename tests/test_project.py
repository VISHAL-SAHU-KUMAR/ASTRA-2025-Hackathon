"""
LunarVision AI - Project Tests
============================

This module contains basic tests to verify the project structure and components.
"""

import os
import sys

def test_project_structure():
    """Test that the project structure is correct"""
    print("Testing project structure...")
    
    # Check that required directories exist
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/models',
        'src',
        'src/preprocessing',
        'src/feature_extraction',
        'src/models',
        'src/visualization',
        'src/web_interface',
        'tests',
        'docs'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"FAIL: Directory '{dir_path}' not found")
            return False
        print(f"PASS: Directory '{dir_path}' exists")
    
    return True

def test_module_imports():
    """Test that modules can be imported"""
    print("\nTesting module imports...")
    
    # Add the src directory to the Python path
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, src_path)
    
    try:
        from src.preprocessing import preprocessor
        print("PASS: Preprocessing module imported successfully")
    except ImportError as e:
        print(f"FAIL: Could not import preprocessing module: {e}")
        return False
    
    try:
        from src.feature_extraction import extractor
        print("PASS: Feature extraction module imported successfully")
    except ImportError as e:
        print(f"FAIL: Could not import feature extraction module: {e}")
        return False
    
    try:
        from src.models import trainer
        print("PASS: Models module imported successfully")
    except ImportError as e:
        print(f"FAIL: Could not import models module: {e}")
        return False
    
    try:
        from src.visualization import visualizer
        print("PASS: Visualization module imported successfully")
    except ImportError as e:
        print(f"FAIL: Could not import visualization module: {e}")
        return False
    
    return True

def test_sample_data_creation():
    """Test that sample data can be created"""
    print("\nTesting sample data creation...")
    
    try:
        # Add the src directory to the Python path
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        sys.path.insert(0, src_path)
        
        from src.preprocessing.preprocessor import create_sample_data
        dummy_path, output_dir = create_sample_data()
        print(f"PASS: Sample data created at {dummy_path}")
        return True
    except Exception as e:
        print(f"FAIL: Could not create sample data: {e}")
        return False

def main():
    """Run all tests"""
    print("LunarVision AI - Project Tests")
    print("=" * 30)
    
    # Change to the project root directory
    project_root = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(project_root)
    
    all_passed = True
    
    # Run tests
    if not test_project_structure():
        all_passed = False
    
    if not test_module_imports():
        all_passed = False
    
    if not test_sample_data_creation():
        all_passed = False
    
    # Print summary
    print("\n" + "=" * 30)
    if all_passed:
        print("All tests PASSED!")
        print("\nProject is ready for development.")
        print("You can now run individual modules:")
        print("  python src/preprocessing/preprocessor.py")
        print("  python src/feature_extraction/extractor.py")
        print("  python src/models/trainer.py")
        print("  python src/visualization/visualizer.py")
        print("  python src/web_interface/app.py")
    else:
        print("Some tests FAILED!")
        print("Please check the errors above and fix them before proceeding.")

if __name__ == "__main__":
    main()