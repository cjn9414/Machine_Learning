% This is the main file for the Semeion Number Recognition Neural Network.
% The data from the dataset is given in the following form:
% 'X' are the pixels for all images in the dataset, unrolled into a vector
% for each individual image.
% 'y' is the classification for each image, described by a matrix, where
% 1 represents the number that is written in the image.
% There are separate X and y values for training the NN and testing the NN.

% Load and display the data

clear; close all; clc;
fprintf("Loading data..\n")
load("semeion.data");
X_train = semeion(1:1300, 1:256);
y_train = semeion(1:1300, 257:end);
X_test = semeion(1301:1593, 1:256);
y_test = semeion(1301:1593, 257:end);
fprintf("Displaying 100 random images taken from the dataset being used for training the neural network.\n")
displayTrainingImages(X_train);
fprintf("Press any button to train the neural network.\n")
pause

% Declare neural network parameters

input_layer_size = size(X_train,2);
hidden_layer_size = 50;
output_layer_size = 10;
lambda = 1;

% Initialize theta values

Theta1 = zeros(hidden_layer_size, input_layer_size + 1); #Add bias unit
Theta2 = zeros(output_layer_size, hidden_layer_size + 1); #Add bias unit

% Randomize values of theta

[init_Theta1, init_Theta2] = initializeWeights(Theta1, Theta2);

% Roll values of theta into a vector in order to pass through fmincg

rolled_weights = [Theta1(:); Theta2(:)];
init_weights = [init_Theta1(:); init_Theta2(:)];

% Fifty iterations to train the neural network
options = optimset('MaxIter', 50);

% Check validity of backpropogation algorithm
debugBackprop(lambda);


% Function to be evaluated and minimized with respect to 't', or the weights of the NN
costFunc = @(t) costFunction(X_train, t, y_train, lambda, hidden_layer_size, input_layer_size, output_layer_size);

% Training the neural network
[rolled_weights cost] = fmincg(costFunc, init_weights, options);

fprintf("Neural network has been trained. Press any button to test the neural network\n")
pause 

% Test the accuracy of the neural network
[accuracy] = predict(X_test, y_test, rolled_weights, hidden_layer_size, input_layer_size, output_layer_size);

fprintf('The percent accuracy of this neural network is approximately: %f\n', accuracy*100)
