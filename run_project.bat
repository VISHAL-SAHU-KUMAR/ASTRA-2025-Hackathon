@echo off
title LunarVision AI - Project Runner

echo ========================================
echo   LunarVision AI - Project Runner
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher and try again
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
pip show opencv-python >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install required packages
        pause
        exit /b 1
    )
)

echo.
echo Select an option:
echo 1. Run complete workflow
echo 2. Run preprocessing module
echo 3. Run feature extraction module
echo 4. Run model training simulation
echo 5. Run visualization module
echo 6. Run web interface
echo 7. Run tests
echo 8. Exit
echo.

choice /c 12345678 /m "Enter your choice"
echo.

if %errorlevel% == 1 (
    echo Running complete workflow...
    python run_all.py
) else if %errorlevel% == 2 (
    echo Running preprocessing module...
    python src/preprocessing/preprocessor.py
) else if %errorlevel% == 3 (
    echo Running feature extraction module...
    python src/feature_extraction/extractor.py
) else if %errorlevel% == 4 (
    echo Running model training simulation...
    python src/models/trainer.py
) else if %errorlevel% == 5 (
    echo Running visualization module...
    python src/visualization/visualizer.py
) else if %errorlevel% == 6 (
    echo Starting web interface...
    echo Visit http://localhost:5000 in your browser
    python src/web_interface/app.py
) else if %errorlevel% == 7 (
    echo Running tests...
    python tests/test_project.py
) else (
    echo Exiting...
    exit /b 0
)

echo.
echo Press any key to continue...
pause >nul