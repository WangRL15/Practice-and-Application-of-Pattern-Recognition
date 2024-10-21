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
