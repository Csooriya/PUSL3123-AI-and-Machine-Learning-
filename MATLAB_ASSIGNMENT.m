% Load the dataset for User 02
data = load('U02_Acc_TimeD_FreqD_FDay.mat');
X = data.Acc_TDFD_Feat_Vec;  % Feature matrix (36 samples x 131 features)

% Create synthetic binary labels for demonstration
Y = mod(1:36, 2);  % Binary labels (alternating 0 and 1)

% Split the data into training (80%) and testing (20%) sets
cv = cvpartition(length(Y), 'HoldOut', 0.2);
X_train = X(cv.training, :);  % Training features
Y_train = Y(cv.training);     % Training labels
X_test = X(cv.test, :);       % Testing features
Y_test = Y(cv.test);          % Testing labels

% Display sizes of the splits
disp(['Training Set: ', num2str(size(X_train, 1)), ' samples']);
disp(['Testing Set: ', num2str(size(X_test, 1)), ' samples']);



% Transpose inputs and targets for the neural network
X_train_nn = X_train';  % Transpose to [features x samples] => [131 x 29]
Y_train_nn = Y_train';  % Transpose to [1 x samples] => [1 x 29]

% Verify dimensions
disp(['Size of X_train_nn: ', num2str(size(X_train_nn))]);
disp(['Size of Y_train_nn: ', num2str(size(Y_train_nn))]);


% Ensure Y_train_nn is transposed correctly
Y_train_nn = Y_train';        % Transpose [29 x 1] to [1 x 29]
Y_train_nn = reshape(Y_train_nn, 1, []);  % Ensure it's a row vector [1 x 29]

% Verify corrected dimensions
disp(['Final Corrected Size of X_train_nn: ', num2str(size(X_train_nn))]);
disp(['Final Corrected Size of Y_train_nn: ', num2str(size(Y_train_nn))]);


% Create the feedforward neural network
hiddenLayerSize = 10;  % Number of neurons in the hidden layer
net = feedforwardnet(hiddenLayerSize, 'traingd');  % Using gradient descent

% Configure training parameters
net.trainParam.lr = 0.01;      % Learning rate
net.trainParam.epochs = 500;  % Number of training epochs
net.trainParam.goal = 1e-5;   % Performance goal

% Train the network
net = train(net, X_train_nn, Y_train_nn);

% Transpose test set for the neural network
X_test_nn = X_test';  % Transpose to [features x samples] => [131 x 7]

% Predict using the trained network
Y_pred = net(X_test_nn);  % Predictions for the test set

% Convert predictions to binary (0 or 1)
Y_pred_binary = round(Y_pred);

% Display predicted labels
disp('Predicted Labels for Test Set:');
disp(Y_pred_binary);


% Calculate accuracy
accuracy = sum(Y_pred_binary == Y_test') / length(Y_test);  % Ensure Y_test is transposed to match dimensions
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Generate and display the confusion matrix
confMat = confusionmat(Y_test, Y_pred_binary');
disp('Confusion Matrix:');
disp(confMat);

% Visualize the confusion matrix
figure;
heatmap(confMat, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'True');

% Plot the actual vs predicted labels
figure;
subplot(2,1,1);
plot(Y_test, 'bo-', 'MarkerFaceColor', 'b');  % Actual labels (blue)
title('Actual Labels');
xlabel('Sample Index');
ylabel('Label');
subplot(2,1,2);
plot(Y_pred_binary', 'ro-', 'MarkerFaceColor', 'r');  % Predicted labels (red)
title('Predicted Labels');
xlabel('Sample Index');
ylabel('Label');



% Perform PCA to reduce dimensionality
[coeff, score, ~, ~, explained] = pca(X_train);

% Choose the number of components to keep (e.g., 95% of the variance)
variance_threshold = 95;  % Retain components explaining 95% variance
cumulative_variance = cumsum(explained);
num_components = find(cumulative_variance >= variance_threshold, 1);

disp(['Number of components to retain: ', num2str(num_components)]);

% Reduce the feature space
X_train_pca = score(:, 1:num_components);  % PCA-reduced training features
X_test_pca = (X_test - mean(X_train)) * coeff(:, 1:num_components);  % Apply PCA to test set

% Create a new feedforward neural network with more neurons
hiddenLayerSize = 20;  % Increased neurons in hidden layer
net = feedforwardnet(hiddenLayerSize, 'trainlm');  % Using Levenberg-Marquardt algorithm

% Update training parameters
net.trainParam.epochs = 1000;  % Increased epochs
net.trainParam.goal = 1e-6;    % Lower performance goal

% Train the network with the PCA-reduced dataset
net = train(net, X_train_pca', Y_train_nn);

% Test the optimized network
Y_pred_optimized = net(X_test_pca');
Y_pred_binary_optimized = round(Y_pred_optimized);

% Evaluate optimized performance
accuracy_optimized = sum(Y_pred_binary_optimized == Y_test') / length(Y_test);
disp(['Optimized Accuracy: ', num2str(accuracy_optimized * 100), '%']);

% Confusion Matrix
confMat_optimized = confusionmat(Y_test, Y_pred_binary_optimized');
disp('Optimized Confusion Matrix:');
disp(confMat_optimized);


% Calculate class weights based on label distribution
classCounts = histcounts(Y_train, unique(Y_train));
weights = max(classCounts) ./ classCounts;

% Assign weights to the network
net.performParam.regularization = 0.1;  % Add regularization
for i = 1:length(weights)
    net.layerWeights{1, i}.delays = weights(i);  % Scale weights by class importance
end

% Train the network again with weights
net = train(net, X_train_pca', Y_train_nn);

% Test the weighted network
Y_pred_weighted = net(X_test_pca');
Y_pred_binary_weighted = round(Y_pred_weighted);

% Evaluate weighted performance
accuracy_weighted = sum(Y_pred_binary_weighted == Y_test') / length(Y_test);
disp(['Weighted Accuracy: ', num2str(accuracy_weighted * 100), '%']);

% Confusion Matrix
confMat_weighted = confusionmat(Y_test, Y_pred_binary_weighted');
disp('Weighted Confusion Matrix:');
disp(confMat_weighted);
