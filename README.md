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
<img width="1500" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/0a86d49e-b0d5-4e7c-9732-736cc3780562" />

<img width="4470" height="1517" alt="sample_images_analysis" src="https://github.com/user-attachments/assets/e92635b7-44c7-453a-ba21-f5d1254a0275" />

<img width="1664" height="1638" alt="demo_confusion_matrix" src="https://github.com/user-attachments/assets/834c66a7-7bfc-4086-8549-0589e9b45752" />

<img width="1664" height="1638" alt="mars_confusion_matrix" src="https://github.com/user-attachments/assets/11b69692-bb79-43a6-bbbd-0d4e351f048f" />

<img width="640" height="480" alt="test" src="https://github.com/user-attachments/assets/06f87710-1869-4446-b9c4-87aa54325316" />


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
## 🚀 Team Dev Coders

**Team Leader:** Vishal Kumar Sahu  
**Challenge:** ASTRA 2025 Hackathon 
**Project:** LunarVision AI – AI-powered water ice detection for Moon and Mars  

We are **Team Dev Coders**, a group of passionate developers and space enthusiasts dedicated to solving critical challenges in space exploration. Our team is led by **Vishal Kumar Sahu**, who brings expertise in AI, machine learning, and planetary science applications.

Together, we are working on **LunarVision AI**, an open-source solution that leverages artificial intelligence to detect water ice from satellite images, enabling sustainable human missions beyond Earth.

## ✅ ASTRA 2025 Hackathon – Where Curiosity Meets Cosmic

We are proud to participate in ASTRA 2025 Hackathon, a platform where innovation, science, and imagination come together to explore the mysteries of space.
“Where Curiosity Meets Cosmic” – this year’s theme encourages participants to push the boundaries of planetary science, resource utilization, and space exploration through technology-driven solutions.

Our project LunarVision AI is our contribution to solving real challenges beyond Earth, aligning with ASTRA’s mission to empower curious minds to build the future of space exploration.

### 💡 Connect with us  
- GitHub: [(https://github.com/VISHAL-SAHU-KUMAR)]  
# ASTRA-2025-Hackathon
>>>>>>> 1f370f1122188185ba9cd9ac97e15c4d3947d35a
