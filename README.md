# 🤖 AI Image Predictions

A sophisticated deep learning model for **content classification** using ResNet-18 architecture. This project can classify images into 6 categories: disturbing content, drug use, horror, neutral, nudity, and violence.

## 📋 Table of Contents
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Contributing](#-contributing)

## 🚀 Features

- **6-Class Content Classification**: Accurately classifies images into predefined categories
- **High Performance**: ResNet-18 based architecture with proven results
- **Easy Testing**: Multiple ways to test single images or entire datasets
- **Comprehensive Analysis**: Detailed performance metrics and confusion matrix
- **User-Friendly**: Simple scripts for both beginners and advanced users

## 🏗️ Model Architecture

- **Base Architecture**: ResNet-18 (18-layer Residual Network)
- **Input Size**: 224×224×3 RGB images
- **Output Classes**: 6 categories
- **Parameters**: ~11.2M trainable parameters
- **Preprocessing**: ImageNet normalization standards

### Class Labels:
1. **disturbing_content** - Disturbing or inappropriate content
2. **drug_use** - Drug-related imagery
3. **horror** - Horror or scary content
4. **neutral** - Safe, neutral content
5. **nudity** - Adult content detection
6. **violence** - Violent or aggressive content

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Step 1: Clone or Download
```bash
# Option 1: Clone with Git
git clone <repository-url>
cd extracted_frames

# Option 2: Download and extract ZIP file
# Navigate to the project folder
cd /Project/<userName>/extracted_frames
```

### Step 2: Install Dependencies
```bash
# Navigate to project directory
cd /Project/<userName>/extracted_frames

# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision Pillow numpy
```

### Step 3: Verify Installation
```bash
# Test basic model loading
python testModal.py
```

## ⚡ Quick Start

### Test a Single Image
```bash
# Direct command for single image testing
python quick_test.py "path/to/your/image.jpg"

# Example:
python quick_test.py "Predict/neutral/frame_0002.jpg"
```

### Test All Images
```bash
# Test all images in the Predict folder
python test_all_images.py
```

## 📖 Usage

### 1. Single Image Testing

Test a single image directly:
```bash
python quick_test.py "Predict/horror/Screenshot 2024-10-13 000341.jpg"
```

Or enter image path interactively:
```bash
python quick_test.py
# Then enter: Predict/neutral/frame_0002.jpg
```

### 2. Batch Testing (All Images)

Test all images in the Predict folder:
```bash
python test_all_images.py
```

This will:
- Test all images in each category subfolder
- Show individual predictions
- Calculate accuracy per category
- Display overall performance metrics
- Show common misclassifications

### 3. Model Verification

Check your model loading and basic info:
```bash
python testModal.py
```

## 📁 Project Structure

```
extracted_frames/
│
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Dependencies
├── 🤖 trained_model.pth         # Pre-trained model weights
│
├── 🧪 Essential Scripts/
│   ├── testModal.py             # Basic model verification
│   ├── quick_test.py            # Single image testing
│   └── test_all_images.py       # Comprehensive batch testing
│
└── 📂 Predict/                  # Test images folder
    ├── disturbing_content/      # Disturbing content samples
    ├── drug_use/               # Drug use samples
    ├── horror/                 # Horror content samples
    ├── neutral/                # Neutral content samples
    ├── nudity/                 # Nudity detection samples
    └── violence/               # Violence detection samples
```

## 🎯 Performance Metrics

Based on comprehensive testing:

| Category | Accuracy | Performance |
|----------|----------|-------------|
| **Horror** | 100.0% | 🔥 Excellent |
| **Neutral** | 100.0% | 🔥 Excellent |
| **Violence** | 100.0% | 🔥 Excellent |
| **Drug Use** | 42.9% | ⚠️ Needs Improvement |
| **Disturbing Content** | 37.5% | ⚠️ Needs Improvement |
| **Nudity** | 15.2% | ❌ Poor |

**Overall Accuracy**: 62.2%

## 🔧 Troubleshooting

### Common Issues:

1. **Model file not found**
   ```
   Error: trained_model.pth not found
   ```
   **Solution**: Ensure `trained_model.pth` is in the project root directory.

2. **Image not found**
   ```
   Error: Image not found!
   ```
   **Solution**: Use relative paths like `Predict/neutral/image.jpg` or absolute paths.

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Generic Path Examples:

```bash
# For different operating systems:

# Windows:
python quick_test.py "C:\Users\<userName>\Project\extracted_frames\Predict\neutral\image.jpg"

# macOS/Linux:
python quick_test.py "/home/<userName>/Project/extracted_frames/Predict/neutral/image.jpg"

# Relative paths (recommended):
python quick_test.py "Predict/neutral/image.jpg"
```

## 🎨 Example Usage

### Testing Multiple Images
```bash
# Example: Test specific images from project directory
cd /Project/<userName>/extracted_frames

# Test multiple images
python quick_test.py "Predict/neutral/frame_0002.jpg"
python quick_test.py "Predict/horror/Screenshot 2024-10-13 000341.jpg"
python quick_test.py "Predict/violence/frame_0671.jpg"
```

### Custom Integration
```python
# You can also import and use the model directly in your code
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
state_dict = torch.load('trained_model.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("your_image.jpg").convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted = torch.max(probabilities, 0)

class_names = ['disturbing_content', 'drug_use', 'horror', 'neutral', 'nudity', 'violence']
result = class_names[predicted.item()]
confidence_score = confidence.item()

print(f"Prediction: {result} (Confidence: {confidence_score:.3f})")
```

## 📊 Model Training Info

- **Architecture**: ResNet-18 with modified final layer
- **Training Data**: Custom content classification dataset
- **Classes**: 6 content categories
- **Input Preprocessing**: Resize to 224×224, ImageNet normalization
- **Framework**: PyTorch

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

**Made with ❤️ for AI-powered content classification**

*For support or questions, please open an issue in the repository.*
