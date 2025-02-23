people = 40;
withinsample = 5;
principlenum = 50;
Row_FACE_Data = [];
Labels = [];

% Load training data
for k = 1:1:people
    for m = 1:2:10
        PathString = ['orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
        ImageData = imread(PathString);

        if size(ImageData, 3) == 3
            ImageData = rgb2gray(ImageData);
        end

        ImageData = double(ImageData);

        if (k == 1 && m == 1)
            [row, col] = size(ImageData);
        end

        RowConcatenate = reshape(ImageData, 1, row * col);
        Row_FACE_Data = [Row_FACE_Data; RowConcatenate];
        Labels = [Labels; k];
    end
end

MeanFace = mean(Row_FACE_Data, 1);
CenteredData = Row_FACE_Data - repmat(MeanFace, size(Row_FACE_Data, 1), 1);
[U, S, V] = svd(CenteredData, "econ");
Eigenfaces = V(:, 1:principlenum);
PCA_Transformed_Data = CenteredData * Eigenfaces;

numClasses = people;
mean_class = zeros(numClasses, size(PCA_Transformed_Data, 2));
S_W = zeros(size(PCA_Transformed_Data, 2));
S_B = zeros(size(PCA_Transformed_Data, 2));

for c = 1:numClasses
    classData = PCA_Transformed_Data(Labels == c, :);
    classMean = mean(classData, 1);
    mean_class(c, :) = classMean;
    classScatter = (classData - repmat(classMean, size(classData, 1), 1))' * ...
                   (classData - repmat(classMean, size(classData, 1), 1));
    S_W = S_W + classScatter;
    mean_diff = (classMean - mean(PCA_Transformed_Data))';
    S_B = S_B + size(classData, 1) * (mean_diff * mean_diff');
end

[W, D] = eig(inv(S_W) * S_B);
[~, sortedIndices] = sort(diag(D), 'descend');
LDA_W = W(:, sortedIndices(1:principlenum));
LDA_Transformed_Data = PCA_Transformed_Data * LDA_W;
LDA_Transformed_Data = LDA_Transformed_Data(:, 1:30);

% Prepare labels for neural network
one_hot_labels = zeros(size(Labels, 1), numClasses);
for i = 1:size(Labels, 1)
    one_hot_labels(i, Labels(i)) = 1;
end

% Initialize Neural Network Parameters
hidden_neurons = 150;
input_neurons = size(LDA_Transformed_Data, 2);
output_neurons = numClasses;
learning_rate = 0.01;
epochs = 300;

% Randomly initialize weights and biases
W1 = randn(input_neurons, hidden_neurons) * 0.01;
B1 = zeros(1, hidden_neurons);
W2 = randn(hidden_neurons, output_neurons) * 0.01;
B2 = zeros(1, output_neurons);

% Initialize array to store training loss
training_loss = zeros(1, epochs);

% Training Loop
for epoch = 1:epochs
    % Forward pass
    Z1 = LDA_Transformed_Data * W1 + B1;
    A1 = 1 ./ (1 + exp(-Z1)); % Sigmoid activation
    Z2 = A1 * W2 + B2;
    A2 = exp(Z2) ./ sum(exp(Z2), 2); % Softmax activation

    % Compute Loss (Cross-Entropy)
    loss = -sum(sum(one_hot_labels .* log(A2))) / size(LDA_Transformed_Data, 1);
    training_loss(epoch) = loss;

    % Backward pass
    dZ2 = A2 - one_hot_labels;
    dW2 = (A1' * dZ2) / size(LDA_Transformed_Data, 1);
    dB2 = sum(dZ2, 1) / size(LDA_Transformed_Data, 1);

    dA1 = dZ2 * W2';
    dZ1 = dA1 .* A1 .* (1 - A1);
    dW1 = (LDA_Transformed_Data' * dZ1) / size(LDA_Transformed_Data, 1);
    dB1 = sum(dZ1, 1) / size(LDA_Transformed_Data, 1);

    % Update weights and biases
    W1 = W1 - learning_rate * dW1;
    B1 = B1 - learning_rate * dB1;
    W2 = W2 - learning_rate * dW2;
    B2 = B2 - learning_rate * dB2;

    % Print loss for every 10 epochs
    if mod(epoch, 10) == 0
        disp(['Epoch ' num2str(epoch) ', Loss: ' num2str(loss)]);
    end
end

% Plot Training Loss
figure;
plot(1:epochs, training_loss, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Training Loss');
title('Training Loss over Epochs');
grid on;

% Testing
correct_count = 0;
total_test_samples = 0;
for k = 1:people
    for m = 1:2:10
        PathString = ['orl3232' '\' num2str(k) '\' num2str(m + 1) '.bmp'];
        ImageData = imread(PathString);

        if size(ImageData, 3) == 3
            ImageData = rgb2gray(ImageData);
        end

        ImageData = double(ImageData);
        RowConcatenate = reshape(ImageData, 1, row * col);
        PCA_TestSample = (RowConcatenate - MeanFace) * Eigenfaces;
        LDA_TestSample = PCA_TestSample * LDA_W;
        LDA_TestSample = LDA_TestSample(:, 1:30);

        % Forward pass for test sample
        Z1 = LDA_TestSample * W1 + B1;
        A1 = 1 ./ (1 + exp(-Z1));
        Z2 = A1 * W2 + B2;
        A2 = exp(Z2) ./ sum(exp(Z2), 2);

        [~, predicted_class] = max(A2, [], 2);

        if predicted_class == k
            correct_count = correct_count + 1;
        end

        total_test_samples = total_test_samples + 1;
    end
end

test_accuracy = (correct_count / total_test_samples) * 100;
disp(['Test Accuracy with BP: ', num2str(test_accuracy), '%']);
