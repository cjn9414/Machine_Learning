%This is the main file for the SVHN Neural Network.A
% 0 is mapped to 10 in y vector
% X is 4D matrix with dimensions as follows:
% Pixels along the 'X' direction
% Pixels along the 'Y' direction
% RGB (0-1)
% Image number

clear; close all; clc;
fprintf("Loading data..\n")
load("semeion.data");
X = semeion(:, 1:256);
y = semeion(:, 257:end);
fprintf("Displaying 100 random images taken from the dataset being used for training the neural network.\n")
displayTrainingImages(X);
fprintf("Press any button to train the neural network.\n")
pause

input_layer_size = size(X,2);
hidden_layer_size = 50;
output_layer_size = 10;
lambda = 1;

Theta1 = zeros(hidden_layer_size, input_layer_size + 1); #Add bias unit
Theta2 = zeros(output_layer_size, hidden_layer_size + 1); #Add bias unit

[init_Theta1, init_Theta2] = initializeWeights(Theta1, Theta2);

rolled_weights = [Theta1(:); Theta2(:)];
init_weights = [init_Theta1(:); init_Theta2(:)];
options = optimset('MaxIter', 50);
debugBackprop(lambda);

costFunc = @(t) costFunction(X, t, y, lambda, hidden_layer_size, input_layer_size, output_layer_size);
[rolled_weights cost] = fmincg(costFunc, init_weights, options);

fprintf("Neural network has been trained. Press any button to test the neural network\n")
pause 

[accuracy] = predict(X, y, rolled_weights, hidden_layer_size, input_layer_size, output_layer_size);

fprintf("The accuracy of this neural network is approximately: \n")
accuracy