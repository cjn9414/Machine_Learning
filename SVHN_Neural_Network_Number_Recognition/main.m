%This is the main file for the SVHN Neural Network.A
% 0 is mapped to 10 in y vector
% X is 4D matrix with dimensions as follows:
% Pixels along the 'X' direction
% Pixels along the 'Y' direction
% RGB (0-1)
% Image number

clear; close all; clc;
fprintf("Loading data..\n")
load("train_32x32.mat");
fprintf("Displaying 100 random images taken from the dataset being used for training the neural network.\n")
displayTrainingImages(X);
fprintf("Press any button to train the neural network.\n")
pause

input_layer_size = size(X,1)*size(X,2)*size(X,3);
%input_layer_size = size(X,1)*size(X,2);
hidden_layer_size = 32;
output_layer_size = 10;
lambda = 1;
X = double(X(:, :, :, 1:1000));
y = y(1:1000, :);


X_unrolled = reshape(permute(X, [1 3 2 4]), size(X, 1)*size(X, 2)*size(X,3), []);
%X_unrolled = reshape([X(:,:,1,:).*.2989 + X(:,:,2,:).*.5870 + X(:,:,3,:).*.1140], size(X, 1)*size(X,2), []);
X_unrolled = [ones(1, size(X_unrolled, 2)); X_unrolled];

Theta1 = zeros(hidden_layer_size, input_layer_size + 1); #Add bias unit
Theta2 = zeros(output_layer_size, hidden_layer_size + 1); #Add bias unit

[init_Theta1, init_Theta2] = initializeWeights(Theta1, Theta2);

rolled_weights = [Theta1(:); Theta2(:)];
init_weights = [init_Theta1(:); init_Theta2(:)];
options = optimset('MaxIter', 75);
debugBackprop(lambda);

costFunc = @(t) costFunction(X_unrolled, t, y, lambda, hidden_layer_size, input_layer_size, output_layer_size);
[rolled_weights cost] = fmincg(costFunc, init_weights, options);

fprintf("Neural network has been trained. Press any button to test the neural network\n")
pause

clear X X_unrolled y
load("test_32x32.mat") 

X_unrolled = reshape(permute(X, [1 3 2 4]), size(X, 1)*size(X, 2)*size(X,3), []);
%X_unrolled = reshape([X(:,:,1,:).*.2989 + X(:,:,2,:).*.5870 + X(:,:,3,:).*.1140], size(X, 1)*size(X,2), []);
X_unrolled = [ones(1, size(X_unrolled, 2)); X_unrolled](:, 1:1000);
y = y(1:1000);

[accuracy] = predict(X_unrolled, y, rolled_weights, hidden_layer_size, input_layer_size, output_layer_size);

fprintf("The accuracy of this neural network is approximately: \n")
accuracy