"""
LunarVision AI - Installation Verification
======================================

This script verifies that all required components are properly installed
and the project is ready for development.
"""

import os
import sys
import importlib

def check_python_version():
    """Check if Python version is sufficient"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - TOO OLD")
        print("  Please upgrade to Python 3.7 or higher")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    required_packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_good = True
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            print(f"✓ {package_name} - INSTALLED")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            print(f"  Install with: pip install {module_name}")
            all_good = False
    
    # Check optional packages
    print("\nChecking optional packages...")
    optional_packages = [
        ("tensorflow", "TensorFlow"),
        ("flask", "Flask")
    ]
    
    for module_name, package_name in optional_packages:
        try:
            importlib.import_module(module_name)
            print(f"✓ {package_name} - INSTALLED")
        except ImportError:
            print(f"○ {package_name} - NOT INSTALLED (optional)")
    
    return all_good

def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_files = [
        "LunarVision_AI_Project_Plan.md",
        "README.md",
        "requirements.txt",
        "run_all.py"
    ]
    
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/models",
        "src",
        "src/preprocessing",
        "src/feature_extraction",
        "src/models",
        "src/visualization",
        "src/web_interface",
        "tests",
        "docs",
        "notebooks"
    ]
    
    all_good = True
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} - EXISTS")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} - EXISTS")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_good = False
    
    return all_good

def check_module_functionality():
    """Check if modules can be imported and run"""
    print("\nChecking module functionality...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    modules_to_test = [
        ("src.preprocessing.preprocessor", "Preprocessing"),
        ("src.feature_extraction.extractor", "Feature Extraction"),
        ("src.models.trainer", "Model Training"),
        ("src.visualization.visualizer", "Visualization")
    ]
    
    all_good = True
    for module_path, module_name in modules_to_test:
        try:
            importlib.import_module(module_path)
            print(f"✓ {module_name} module - IMPORT SUCCESS")
        except ImportError as e:
            print(f"✗ {module_name} module - IMPORT FAILED: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all verification checks"""
    print("LunarVision AI - Installation Verification")
    print("=" * 45)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Module Functionality", check_module_functionality)
    ]
    
    results = []
    for check_name, check_function in checks:
        print(f"\n{check_name}")
        print("-" * len(check_name))
        result = check_function()
        results.append((check_name, result))
    
    # Summary
    print("\n" + "=" * 45)
    print("VERIFICATION SUMMARY")
    print("=" * 45)
    
    all_passed = True
    for check_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYour LunarVision AI project is ready for development!")
        print("\nNext steps:")
        print("1. Run individual modules:")
        print("   python src/preprocessing/preprocessor.py")
        print("   python src/feature_extraction/extractor.py")
        print("   python src/models/trainer.py")
        print("   python src/visualization/visualizer.py")
        print("\n2. Run the complete workflow:")
        print("   python run_all.py")
        print("\n3. Try the web interface:")
        print("   python src/web_interface/app.py")
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("\nPlease address the issues above before proceeding.")
        print("Refer to the project documentation for installation instructions.")

if __name__ == "__main__":
    main()