# **Deep Transfer Learning (ResNet-152) with Comprehensive Image Pre-processing for Pneumonia Classification in Chest Radiographs**

## 📋 **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🔬 **Overview**

This repository contains the complete implementation of a deep transfer learning system for automated pneumonia detection from chest X-ray images. The system employs ResNet-152 architecture with comprehensive image pre-processing techniques to achieve high-accuracy binary classification between pneumonia and normal cases.

### **Key Highlights:**
- 🎯 **98.86% Training Accuracy** and **82.46% Test Accuracy**
- 🏥 **Clinical-grade Performance** with interpretable results
- 📊 **Comprehensive Evaluation** with statistical significance testing
- 🔍 **Visual Explanations** through Class Activation Maps (CAM)
- ⚡ **Production-ready** deployment scripts

### **Research Paper:**
The complete implementation of this deep transfer learning system, including ResNet-152 source code, comprehensive image pre-processing scripts, and experimental frameworks, is available at: paper/deep_transfer_learning_pneumonia_resnet152.pdf

## ✨ **Features**

### **Core Capabilities:**
- **Deep Transfer Learning** with ResNet-152 pre-trained on ImageNet
- **Comprehensive Image Pre-processing** pipeline
- **Advanced Data Augmentation** strategies
- **Class Activation Maps** for interpretability
- **Cross-validation** support
- **Statistical Analysis** tools
- **Performance Visualization** suite
- **Model Deployment** utilities

### **Technical Features:**
- **PyTorch Framework** implementation
- **GPU Acceleration** support
- **Configurable Hyperparameters**
- **Automated Model Checkpointing**
- **Tensorboard Integration**
- **Docker Containerization**
- **REST API** for inference

## 📊 **Dataset**

### **Data Sources:**
1. **RSNA Pneumonia Detection Challenge** (Kaggle)
   - Deidentified chest X-ray images
   - Binary labels: Pneumonia/Normal
   - Multi-institutional collection

2. **COVID-19 Image Data Collection** (University of Montreal)
   - International COVID-19 chest X-rays
   - Research-grade annotations
   - Diverse patient demographics

### **Dataset Statistics:**
```
Total Images: 3,080
├── Training Set: 2,624 images (85.2%)
├── Validation Set: 228 images (7.4%)
└── Test Set: 228 images (7.4%)

Class Distribution:
├── Pneumonia Cases: 1,540 images (50%)
└── Normal Cases: 1,540 images (50%)
```

## 🏗 **Architecture**

### **Model Overview:**
```
ResNet-152 (Pre-trained ImageNet)
├── Convolutional Layers (Frozen)
├── Residual Blocks (152 layers)
├── Global Average Pooling
├── Custom Classifier
│   ├── Linear Layer (2048 → 2)
│   └── LogSoftmax Activation
└── Class Activation Maps
```

### **Transfer Learning Strategy:**
- **Feature Extraction:** Freeze pre-trained layers
- **Fine-tuning:** Only final classification layer
- **Optimization:** Adam optimizer with NLLLoss
- **Regularization:** Global Average Pooling

## 🚀 **Installation**

### **Requirements:**
```bash
Python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### **Quick Setup:**
```bash
# Clone repository
git clone https://github.com/furkanhanilci/DTL_with_Comprehensive_Image_Processing_for_Pneumonia_Classification_in_Chest_Radiographs.git
cd DTL_with_Comprehensive_Image_Processing_for_Pneumonia_Classification_in_Chest_Radiographs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Docker Installation:**
```bash
# Build Docker image
docker build -t dtl-pneumonia-classifier .

# Run container
docker run -p 8000:8000 --gpus all dtl-pneumonia-classifier
```

## 📝 **Usage**

### **Quick Start:**
```python
from src.model import DTLPneumoniaClassifier
from src.preprocessing import ImagePreprocessor

# Initialize model
model = DTLPneumoniaClassifier.load_pretrained('models/best_model.pth')

# Preprocess image
preprocessor = ImagePreprocessor()
image = preprocessor.process('path/to/chest_xray.jpg')

# Make prediction
prediction, confidence = model.predict(image)
cam = model.generate_cam(image)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")
```

### **Command Line Interface:**
```bash
# Single image prediction
python predict.py --image path/to/image.jpg --model models/best_model.pth

# Batch prediction
python predict.py --batch --input_dir data/test/ --output results.csv

# Generate CAM visualization
python visualize.py --image path/to/image.jpg --save_cam cam_output.jpg
```

## 🎯 **Model Training**

### **Training Configuration:**
```yaml
# config/train_config.yaml
model:
  architecture: resnet152
  pretrained: true
  num_classes: 2

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  optimizer: adam
  scheduler: steplr
  
data:
  image_size: 224
  augmentation: true
  normalization: imagenet
```

### **Training Commands:**
```bash
# Start training
python train.py --config config/train_config.yaml

# Resume training
python train.py --resume checkpoints/epoch_50.pth

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### **Training Monitoring:**
```bash
# Launch Tensorboard
tensorboard --logdir logs/

# Monitor training progress
python monitor.py --log_dir logs/training/
```

## 📈 **Evaluation**

### **Evaluation Metrics:**
```bash
# Comprehensive evaluation
python evaluate.py --model models/best_model.pth --test_data data/test/

# Cross-validation
python cross_validate.py --folds 5 --data data/

# Statistical analysis
python statistics.py --results results/ --significance_test
```

### **Performance Metrics:**
```
Classification Metrics:
├── Accuracy: 82.46% ± 2.1%
├── Precision: 83.2% ± 1.8%
├── Recall: 80.7% ± 2.3%
├── F1-Score: 81.9% ± 1.9%
├── Specificity: 84.2% ± 2.0%
├── AUC-ROC: 0.891 ± 0.023
└── Cohen's Kappa: 0.649 ± 0.041

Computational Metrics:
├── Training Time: 4.2 hours
├── Inference Time: 23ms per image
├── Memory Usage: 8.4GB GPU
└── Model Size: 232MB
```

## 📊 **Results**

### **Performance Comparison:**
| Method | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| **Our Method (ResNet-152)** | **82.46%** | **83.2%** | **80.7%** | **81.9%** | **0.891** |
| CheXNet | 76.8% | 78.1% | 75.2% | 76.6% | 0.863 |
| DenseNet-121 | 79.3% | 80.5% | 77.8% | 79.1% | 0.875 |
| ResNet-50 | 78.9% | 79.7% | 77.1% | 78.4% | 0.869 |

### **Clinical Impact:**
- 🕒 **Diagnosis Time Reduction:** From 11 days to 3 days
- 👩‍⚕️ **Radiologist Support:** Visual attention guidance
- 🏥 **Workflow Integration:** PACS compatibility
- 📈 **Throughput Improvement:** 95% faster initial screening

## 🔍 **Visualization**

### **Class Activation Maps:**
```python
# Generate CAM visualizations
python visualize_cam.py --model models/best_model.pth --images data/samples/

# Interactive visualization
python app.py --port 8080
```

### **Training Visualizations:**
```python
# Plot training curves
python plot_training.py --log_file logs/training.log

# Generate confusion matrix
python plot_confusion_matrix.py --predictions results/predictions.csv
```

## 📁 **Project Structure**

```
DTL_with_Comprehensive_Image_Processing_for_Pneumonia_Classification_in_Chest_Radiographs/
├── README.md
├── requirements.txt
├── Dockerfile
├── setup.py
├── config/
│   ├── train_config.yaml
│   ├── data_config.yaml
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── resnet152.py
│   │   ├── transfer_learning.py
│   │   └── cam_generator.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── validator.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── checkpoint.py
│       └── config_parser.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize.py
│   └── cross_validate.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── results_analysis.ipynb
│   └── visualization_examples.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── best_model.pth
│   ├── checkpoint_epoch_50.pth
│   └── model_weights/
├── results/
│   ├── metrics/
│   ├── visualizations/
│   ├── cam_outputs/
│   └── reports/
├── logs/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── docs/
│   ├── installation.md
│   ├── usage_guide.md
│   ├── api_reference.md
│   └── paper.pdf
└── tests/
    ├── test_model.py
    ├── test_preprocessing.py
    ├── test_training.py
    └── test_evaluation.py
```

## ⚙️ **Configuration**

### **Model Configuration:**
```yaml
# config/model_config.yaml
architecture:
  name: resnet152
  pretrained: true
  freeze_backbone: true
  
classifier:
  input_features: 2048
  num_classes: 2
  activation: logsoftmax
  
cam:
  enabled: true
  layer: layer4
  upsampling_mode: bilinear
```

### **Training Configuration:**
```yaml
# config/train_config.yaml
optimizer:
  name: adam
  lr: 0.001
  weight_decay: 1e-4
  
scheduler:
  name: steplr
  step_size: 4
  gamma: 0.1
  
early_stopping:
  patience: 10
  min_delta: 0.001
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup:**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/
```

## 📄 **Citation**

If you use this work in your research, please cite:

```bibtex
@article{hanilci2025deep,
  title={Deep Transfer Learning (ResNet-152) with Comprehensive Image Pre-processing for Pneumonia Classification in Chest Radiographs},
  author={Hanilci, Furkan},
  journal={},
  year={},
  publisher={Furkan Hanilçi},
  url={https://github.com/furkanhanilci/DTL_with_Comprehensive_Image_Processing_for_Pneumonia_Classification_in_Chest_Radiographs}
}
```

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Contact**

**Furkan HANİLÇİ**
- 🏢 KARSAN Automotive R&D Department & Uludag University Electrical Electronics Engineering
- 📍 Bursa, Turkey
- 📧 Email: furkan.hanilci@karsan.com.tr & 502305019@ogr.uludag.edu.tr
- 🐙 GitHub: [@furkanhanilci](https://github.com/furkanhanilci)

## 🙏 **Acknowledgments**

- **RSNA** for providing the Pneumonia Detection Challenge dataset
- **University of Montreal** for the COVID-19 image collection
- **PyTorch Team** for the deep learning framework
- **ImageNet** for pre-trained model weights

---

**⭐ If you find this project useful, please consider giving it a star!**
