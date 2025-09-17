# LunarVision AI - User Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Using the Modules](#using-the-modules)
5. [Running the Complete Workflow](#running-the-complete-workflow)
6. [Web Interface](#web-interface)
7. [Testing](#testing)
8. [Contributing](#contributing)

## Project Overview

LunarVision AI is an AI-powered system designed to detect water ice on the surfaces of the Moon and Mars using remote sensing satellite images. The system provides a complete pipeline from data preprocessing to visualization of results.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installing Dependencies

1. Clone or download the project repository
2. Navigate to the project directory
3. Install the required packages:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- OpenCV for image processing
- NumPy for numerical computations
- TensorFlow for machine learning
- Matplotlib and Plotly for visualization
- Flask for the web interface

## Project Structure

The project is organized into the following directories:

```
lunarvision_ai/
├── data/
│   ├── raw/          # Raw satellite images
│   ├── processed/    # Preprocessed images and results
│   └── models/       # Trained models
├── src/
│   ├── preprocessing/     # Image preprocessing module
│   ├── feature_extraction/ # Feature extraction module
│   ├── models/            # Machine learning models
│   ├── visualization/     # Visualization tools
│   └── web_interface/     # Web application
├── notebooks/        # Jupyter notebooks for demonstration
├── tests/            # Test scripts
├── docs/             # Documentation
├── requirements.txt  # Python dependencies
└── README.md         # Project overview
```

## Using the Modules

### 1. Preprocessing Module

The preprocessing module handles image enhancement and noise reduction.

```bash
python src/preprocessing/preprocessor.py
```

This will:
- Create sample data in `data/raw/`
- Apply noise reduction, brightness enhancement, and shadow correction
- Save processed images in `data/processed/`

### 2. Feature Extraction Module

The feature extraction module identifies relevant characteristics in images.

```bash
python src/feature_extraction/extractor.py
```

This will:
- Extract edges using Canny edge detection
- Calculate color histograms
- Compute texture features using gradients
- Prepare feature vectors for machine learning

### 3. Models Module

The models module handles training of machine learning models.

```bash
python src/models/trainer.py
```

This will:
- Define a CNN architecture for ice detection
- Create a dummy dataset for demonstration
- Simulate model training
- Save the model in `data/models/`

### 4. Visualization Module

The visualization module generates heatmaps and other visualizations.

```bash
python src/visualization/visualizer.py
```

This will:
- Generate heatmaps showing ice detection confidence
- Plot confidence score distributions
- Create 3D terrain visualizations

## Running the Complete Workflow

To run the complete workflow, you can use the Jupyter notebook:

```bash
jupyter notebook notebooks/workflow_demo.ipynb
```

Or run each module sequentially as shown in the [Using the Modules](#using-the-modules) section.

## Web Interface

The project includes a web interface for easy interaction with the system.

```bash
python src/web_interface/app.py
```

This will:
- Start a Flask web server on `http://localhost:5000`
- Provide a user interface for uploading satellite images
- Display analysis results and generate reports

To use the web interface:
1. Visit `http://localhost:5000` in your browser
2. Upload a satellite image
3. View the analysis results
4. Download PDF reports

## Testing

To verify that the project is set up correctly, run the test suite:

```bash
python tests/test_project.py
```

This will:
- Check that all required directories exist
- Verify that modules can be imported
- Test sample data creation

## Contributing

We welcome contributions to the LunarVision AI project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

Please follow the existing code style and include documentation for any new features.