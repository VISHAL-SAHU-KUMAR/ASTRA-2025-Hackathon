#!/bin/bash

# LunarVision AI - Project Runner
# ========================================

echo "========================================"
echo "  LunarVision AI - Project Runner"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.7 or higher and try again"
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
if ! python3 -c "import cv2" &> /dev/null
then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install required packages"
        exit 1
    fi
fi

echo
echo "Select an option:"
echo "1. Run complete workflow"
echo "2. Run preprocessing module"
echo "3. Run feature extraction module"
echo "4. Run model training simulation"
echo "5. Run visualization module"
echo "6. Run web interface"
echo "7. Run tests"
echo "8. Exit"
echo

read -p "Enter your choice (1-8): " choice

echo

case $choice in
    1)
        echo "Running complete workflow..."
        python3 run_all.py
        ;;
    2)
        echo "Running preprocessing module..."
        python3 src/preprocessing/preprocessor.py
        ;;
    3)
        echo "Running feature extraction module..."
        python3 src/feature_extraction/extractor.py
        ;;
    4)
        echo "Running model training simulation..."
        python3 src/models/trainer.py
        ;;
    5)
        echo "Running visualization module..."
        python3 src/visualization/visualizer.py
        ;;
    6)
        echo "Starting web interface..."
        echo "Visit http://localhost:5000 in your browser"
        python3 src/web_interface/app.py
        ;;
    7)
        echo "Running tests..."
        python3 tests/test_project.py
        ;;
    8)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo
echo "Press Enter to continue..."
read