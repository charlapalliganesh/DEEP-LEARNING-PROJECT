# DEEP-LEARNING-PROJECT
COMPANY : CODTECH IT SOLUTIONS NAME :CHARLAPALLI GANESH INTERN ID : CT08DF453 DOMAIN : DATA SCIENCE DURATION : 8 WEEKS MENTOR : NEELA SANTHOSH KUMAR

# CIFAR-10 Image Classification with CNN (PyTorch)

This project implements an image classification model using a **Convolutional Neural Network (CNN)** on the **CIFAR-10 dataset** with PyTorch. It includes data loading, model building, training, validation, and performance visualization.

---

##  Project Structure
```
├── Deep learning project .ipynb
├── README.md
```

---

##  Project Overview

- **Objective:** To classify images into one of 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a CNN model.
- **Dataset:** CIFAR-10, consisting of 60,000 32x32 color images across 10 classes (50,000 training and 10,000 test images).
- **Framework:** PyTorch

---

## Features

- Data loading with torchvision's CIFAR-10 dataset.
- Data normalization for better model convergence.
- Custom CNN with 3 convolutional layers and fully connected layers.
- Accuracy evaluation on both training and validation data.
- Accuracy visualization across epochs.

---

##  Model Architecture

1. **Convolutional Layer 1:** 3 input channels → 32 filters
2. **ReLU Activation**
3. **MaxPooling Layer**
4. **Convolutional Layer 2:** 32 → 64 filters
5. **ReLU Activation**
6. **MaxPooling Layer**
7. **Convolutional Layer 3:** 64 → 64 filters
8. **ReLU Activation**
9. **Fully Connected Layer 1:** Flattened output → 64 units
10. **ReLU Activation**
11. **Fully Connected Layer 2:** 64 → 10 units (for 10 classes)

---

##  Training & Results

- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10
- **Batch Size:** 64
- Final accuracy is printed, and a plot of training vs. validation accuracy is displayed.

---

## 🛠 Setup Instructions

### Prerequisites
- Python 3.x
- pip

### Install Dependencies
```bash
pip install torch torchvision torchaudio matplotlib
```

### Running the Notebook
1. Open the `Deep learning project .ipynb` in Jupyter Notebook or JupyterLab.
2. Execute all the cells to train and evaluate the model.

---

## 🖼️ Output

- Prints training and validation accuracy per epoch.
- Displays a graph of training vs validation accuracy.
- Final test accuracy is printed at the end.

---

##  Future Improvements

- Implement data augmentation for better generalization.
- Tune hyperparameters like learning rate, batch size, number of epochs.
- Explore deeper architectures or pre-trained models like ResNet.

---
AUTHOR
CHARLAPALLI GANESH
