"""
LunarVision AI - Complete Workflow Runner
======================================

This script runs the complete LunarVision AI workflow:
1. Data preprocessing
2. Feature extraction
3. Model training simulation
4. Visualization
"""

import os
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_preprocessing():
    """Run the preprocessing module"""
    print("=" * 50)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 50)
    
    try:
        from src.preprocessing.preprocessor import main as preprocess_main
        preprocess_main()
        print("✓ Preprocessing completed successfully\n")
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}\n")
        return False

def run_feature_extraction():
    """Run the feature extraction module"""
    print("=" * 50)
    print("STEP 2: FEATURE EXTRACTION")
    print("=" * 50)
    
    try:
        from src.feature_extraction.extractor import main as extract_main
        extract_main()
        print("✓ Feature extraction completed successfully\n")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}\n")
        return False

def run_model_training():
    """Run the model training simulation"""
    print("=" * 50)
    print("STEP 3: MODEL TRAINING SIMULATION")
    print("=" * 50)
    
    try:
        from src.models.trainer import main as train_main
        train_main()
        print("✓ Model training simulation completed successfully\n")
        return True
    except Exception as e:
        print(f"✗ Model training simulation failed: {e}\n")
        return False

def run_visualization():
    """Run the visualization module"""
    print("=" * 50)
    print("STEP 4: VISUALIZATION")
    print("=" * 50)
    
    try:
        from src.visualization.visualizer import main as visualize_main
        visualize_main()
        print("✓ Visualization completed successfully\n")
        return True
    except Exception as e:
        print(f"✗ Visualization failed: {e}\n")
        return False

def main():
    """Run the complete LunarVision AI workflow"""
    print("LunarVision AI - Complete Workflow Runner")
    print("=" * 45)
    print("Starting the complete ice detection workflow...\n")
    
    # Track execution time
    start_time = time.time()
    
    # Run all steps
    steps = [
        ("Data Preprocessing", run_preprocessing),
        ("Feature Extraction", run_feature_extraction),
        ("Model Training Simulation", run_model_training),
        ("Visualization", run_visualization)
    ]
    
    results = []
    for step_name, step_function in steps:
        print(f"Starting {step_name}...")
        result = step_function()
        results.append((step_name, result))
        time.sleep(1)  # Brief pause between steps
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print summary
    print("=" * 50)
    print("WORKFLOW SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for step_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{step_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    if all_passed:
        print("\n🎉 All steps completed successfully!")
        print("\nNext steps:")
        print("1. Explore the results in the 'data/processed/' directory")
        print("2. Run the web interface: python src/web_interface/app.py")
        print("3. Check the Jupyter notebook: jupyter notebooks/workflow_demo.ipynb")
    else:
        print("\n⚠️  Some steps failed. Please check the error messages above.")

if __name__ == "__main__":
    main()