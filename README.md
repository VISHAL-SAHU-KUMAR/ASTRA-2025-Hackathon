<<<<<<< HEAD
# LunarVision AI - COMPLETED

An AI-powered system for detecting water ice on the surfaces of the Moon and Mars using remote sensing satellite images.

## Project Status: ✅ COMPLETED SUCCESSFULLY

LunarVision AI is designed to analyze satellite imagery to identify potential water ice deposits on lunar and Martian surfaces. This system leverages machine learning techniques to process remote sensing data and provide accurate ice detection capabilities that can support future space missions and in-situ resource utilization.

## Key Features

- Image preprocessing and enhancement
- Feature extraction for ice detection
- Machine learning classification (CNN-based)
- Unsupervised clustering for unlabeled data
- Visualization and heatmap generation
- Web-based user interface
- Automated report generation

## Directory Structure

```
lunarvision_ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── models/
│   ├── visualization/
│   └── web_interface/
├── notebooks/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd lunarvision_ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The project is organized into modules that can be run independently:

1. **Data Preprocessing**: Clean and enhance satellite images
2. **Feature Extraction**: Identify relevant characteristics in images
3. **Model Training**: Train ice detection models (multiple approaches implemented)
4. **Visualization**: Generate heatmaps and reports
5. **Web Interface**: Run the Flask-based UI with attractive dashboard

### Quick Start

For Windows:
```
run_project.bat
```

For Linux/Mac:
```
./run_project.sh
```

Or run the complete workflow directly:
```
python run_all.py
```

### Individual Module Execution

```bash
# Run preprocessing
python src/preprocessing/preprocessor.py

# Run feature extraction
python src/feature_extraction/extractor.py

# Run basic model training simulation
python src/models/trainer.py

# Run advanced model training (with TensorFlow if available)
python src/models/advanced_trainer.py

# Run enhanced PM4W model training (scientifically validated approach)
python src/models/enhanced_trainer.py

# Run visualization
python src/visualization/visualizer.py

# Run web interface
python src/web_interface/app.py
```

## Project Completion Status

✅ **All components successfully implemented and tested**
✅ **Scientifically validated PM4W method implemented**
✅ **Professional web interface with attractive dashboard**
✅ **Comprehensive documentation and visualization outputs**
✅ **Ready for real satellite data integration**

## Contributing

We welcome contributions to the LunarVision AI project. Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Planetary Data System
- European Space Agency
- Open-source astronomy community
=======
# ASTRA-2025-Hackathon
>>>>>>> 1f370f1122188185ba9cd9ac97e15c4d3947d35a
