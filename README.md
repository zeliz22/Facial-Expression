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


## first model
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
