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
├── Conv Block 1: 32 → 32 filters, MaxPool, Dropout(0.25)
├── Conv Block 2: 64 → 64 filters, MaxPool, Dropout(0.25)
├── Conv Block 3: 128 → 128 filters, MaxPool, Dropout(0.25)
├── Fully Connected: 512 units, Dropout(0.5)
├── Fully Connected: 256 units, Dropout(0.5)
└── Output: 7 classes (softmax)


### Hyperparameters
Optimizer: Adam (lr=0.001, weight_decay=1e-4)
Loss Function: CrossEntropyLoss
Batch Size: 64
Epochs: 25
Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
### Results
Final Validation Accuracy: 62.03%
Final Validation Loss: 1.00
Final Training Accuracy: 60.90%
Final Train Loss: 1.05
