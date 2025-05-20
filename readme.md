## Sushify: Modular PyTorch Image Classifier

A clean, modular PyTorch project for training and evaluating EfficientNet-based models on a custom sushi image dataset.

---

### Table of Contents

1. [Project Overview](#project-overview)  
2. [Installation & Setup](#installation--setup)  
---

### Project Overview

Sushiify demonstrates a modular approach to building, training, and deploying image classification models using PyTorch and torchvision’s EfficientNet variants. The codebase is organized to separate data handling, model definition, training engine, and prediction/visualization logic.

Key features:
- **Pretrained EfficientNet** (B0 & B2) with transfer learning  
- **Modular data loaders** for train/test splits with torchvision transforms  
- **Reusable training engine** with logging via TensorBoard  
- **Utility functions** for reproducibility and loss-curve plotting  
- **Inference scripts** for batch and custom-image prediction with visualization  

---

### Installation & Setup
**Clone the repo**  
```bash
git clone https://github.com/LorenzoFabbri/Sushiify.git
cd Sushiify
```
**Create a Python environment (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # on macOS/Linux
```


**Install required libraries**
```bash
pip install torch torchvision torchinfo tensorboard tqdm pillow matplotlib requests
```