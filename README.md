# Practice-and-Application-of-Pattern-Recognition
## Midterm Project
### HW1
This code implements a simple feedforward neural network that uses randomly initialized weights for training to predict target values in the Iris dataset. The neural network consists of one hidden layer and one output layer, using the logarithmic sigmoid (logsig) activation function and a linear activation function (purelin).

1. **Input Data**: `iris_in.csv` and `iris_out.csv` represent the input features and target values from the Iris dataset.
2. **Weight Initialization**: Random weights for the hidden and output layers are initialized within the range of [-1, 1].
3. **Training Process**: 
   - During each forward pass, the network computes the hidden layer outputs and the final output, then calculates the error (difference between true values and predicted values).
   - Backpropagation is used to update the weights.
   - The network undergoes 100 epochs of training, calculating the root mean square error (RMSE) after each epoch.
4. **Testing Phase**: The last 75 samples are used for testing, and the accuracy of the predicted results is evaluated against the actual results.
5. **Results**: The final RMSE is output after training, along with the accuracy of the testing results.

### HW2
This MATLAB script implements a feedforward neural network to classify the well-known Iris dataset. The script includes the following key features:

1. **Data Preparation**: Reads the input and output data from CSV files (iris_in.csv and iris_out.csv), and converts the class labels into one-hot encoding.
2. **Network Architecture**: The network consists of a hidden layer with 10 neurons and an output layer with 3 neurons. The hidden layer uses the logsig activation function, and the output layer uses the purelin activation function.
3. **Training Process**:  The network is trained using backpropagation with a learning rate (alpha) of 0.1 over 100 epochs. The weights and biases are randomly initialized, and the   mean squared error (MSE) is computed at each epoch.
4. **Testing**: After training, the network is tested on a separate test set, and the classification accuracy is calculated.
Visualization: The script plots the root mean square error (RMSE) over epochs to track the training progress.

## Final Project
Face Recognition Using PCA and Entropy-based BackPropagation on the ORL Dataset

### Overview

This project implements a face recognition model using the ORL dataset with Principal Component Analysis (PCA) for dimensionality reduction, MaxMin normalization for data preprocessing, and an Entropy-based BackPropagation algorithm for training. The model is designed to achieve at least 85% accuracy on the testing data.
Process

### Technical Implementation

**PCA (Principal Component Analysis)**: Used to reduce the dimensionality of the dataset while retaining as much variance as possible.
MaxMin Normalization: This method scales the data so that it falls within a specified range, ensuring that all input features have the same scale, improving the performance of the neural network.

**Entropy-based BackPropagation**: A variant of the traditional backpropagation algorithm, using entropy to guide the optimization process during training.
Data Split:
**Training Data**: 200 samples (5 samples per class selected from the 1st, 3rd, 5th, 7th, and 9th images of each class in the ORL dataset).
**Testing Data**: 200 samples, with an expected accuracy of at least 85%.

### Network Architecture

Number of Hidden Layers: 1
Number of Neurons in Hidden Layer: 150
Learning Rate: 0.01
Training Epochs: 300

### Objective

The goal is to use the above configuration to train a neural network model that can accurately classify faces from the ORL dataset with a high accuracy on the testing data.
