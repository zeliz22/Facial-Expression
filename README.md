# Facial-Expression
### Overview
This project implements a Convolutional Neural Network (CNN) for facial expression recognition using PyTorch. The model classifies facial expressions into 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
### DataSet
Source: FER2013 dataset from Kaggle's "Challenges in Representation Learning: Facial Expression Recognition Challenge"
Image Format: 48x48 pixel grayscale images
Training Samples: 28,709 images
Test Samples: 3,589 images (public leaderboard)
Classes: 7 emotion categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

Training Set: 80%
Validation Set: 20%


## Bacis model
### CNN Structure 
Input: 48x48x1 grayscale images
├── Conv Block 1: 1 → 32 filters, Kernel=3x3, ReLU, MaxPool(2x2)
├── Conv Block 2: 32 → 64 filters, Kernel=3x3, ReLU, MaxPool(2x2)
├── Conv Block 3: 64 → 128 filters, Kernel=3x3, ReLU, MaxPool(2x2)
├── Fully Connected: 512 units, ReLU, Dropout(0.5)
└── Output: 7 classes (logits)

### Hyperparameters
Batch Size: 64
Epochs: 15

### Results
Final Validation Accuracy: 0.57
Final Validation Loss: 0.6
Final Training Accuracy: 0.59
Final Train Loss: 0.6

## BasicModel+BatchNormalization
### CNN Structure
├── Conv Block 1:
│   ├── Conv2D(1 → 64), BatchNorm, ReLU
│   └── Conv2D(64 → 64), BatchNorm, ReLU, MaxPool(2x2)
├── Conv Block 2:
│   ├── Conv2D(64 → 128), BatchNorm, ReLU
│   └── Conv2D(128 → 128), BatchNorm, ReLU, MaxPool(2x2)
├── Conv Block 3:
│   ├── Conv2D(128 → 256), BatchNorm, ReLU
│   └── Conv2D(256 → 256), BatchNorm, ReLU, MaxPool(2x2)
├── Conv Block 4:
│   └── Conv2D(256 → 512), BatchNorm, ReLU, GlobalAvgPool(1x1)
├── Fully Connected:
│   ├── Linear(512 → 256), ReLU, Dropout(0.5)
│   ├── Linear(256 → 128), ReLU, Dropout(0.5)
│   └── Linear(128 → 7) → Output logits

### Hyperparameters
Batch Size: 64
Epochs: 15
Optimizer: Adam (lr=0.001)
Loss Function: CrossEntropyLoss
Dropout: 0.5 in both fully connected layers

### Results
Final Training Accuracy: 0.90
Final Validation Accuracy: 0.60
Final Training Loss:  0.26
Final Validation Loss:  1.8

The model achieved a very high training accuracy (~90%) but only a moderate validation accuracy 60%, indicating overfitting.
why?????(TODO: kargad ver vxvdebi jer mizezs, unda davwero mere ram gamoiwvia es) maybe model is deeper and more powerful than the dataset can fully utilize, so it fits training data very well but fails to generalize.

## BaseModel+Earyly Stopping
I added data augmentation for generalization and early stopping to prevent overfitting(and Residual Blocks)
# CNN Structure
├── Initial Block:
│ ├── Conv2D(1 → 64, kernel=7, stride=2, padding=3)
│ ├── BatchNorm2D(64)
│ ├── ReLU
│ └── MaxPool2D(kernel=3, stride=2, padding=1)
├── Residual Block Layer 1 (64 → 64, stride=1)
│ ├── ResidualBlock × 2
│ │ ├── Conv2D(64 → 64), BatchNorm, ReLU
│ │ ├── Conv2D(64 → 64), BatchNorm
│ │ └── Identity shortcut
├── Residual Block Layer 2 (64 → 128, stride=2)
│ ├── ResidualBlock × 2
│ │ ├── Conv2D(64/128 → 128), BatchNorm, ReLU
│ │ ├── Conv2D(128 → 128), BatchNorm
│ │ └── Shortcut: Conv2D(64 → 128, kernel=1, stride=2), BatchNorm
├── Residual Block Layer 3 (128 → 256, stride=2)
│ ├── ResidualBlock × 2
│ │ ├── Conv2D(128/256 → 256), BatchNorm, ReLU
│ │ ├── Conv2D(256 → 256), BatchNorm
│ │ └── Shortcut: Conv2D(128 → 256, kernel=1, stride=2), BatchNorm
├── Residual Block Layer 4 (256 → 512, stride=2)
│ ├── ResidualBlock × 2
│ │ ├── Conv2D(256/512 → 512), BatchNorm, ReLU
│ │ ├── Conv2D(512 → 512), BatchNorm
│ │ └── Shortcut: Conv2D(256 → 512, kernel=1, stride=2), BatchNorm

### Hyperparameters
Batch Size: 64
Epochs: 15
Optimizer: Adam (learning rate = 0.001)
Loss Function: CrossEntropyLoss
Dropout: 0.5 before the final fully connected layer
Data Augmentation:
RandomHorizontalFlip(p=0.5)
RandomRotation(±10°)
RandomAffine(translate=(0.1, 0.1))
Input Image Size: 48×48 grayscale
Normalization: Scaled pixel values to [0, 1] range (ToTensor())

### Results
Final Training Accuracy: 0.67
Final Validation Accuracy: 0.63
Final Training Loss: 0.86
Final Validation Loss: 1

Adding early stopping help us to stop overfitting. also, building more complex CNN help to get higher val accurace. maybe it is not a big jump from 0.6 to 0.63 but at least, it was clearly overfitting which is not anymore. 

