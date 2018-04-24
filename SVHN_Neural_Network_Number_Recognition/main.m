%This is the main file for the SVHN Neural Network.
% 0 is mapped to 10 in y vector
% X is 4D matrix with dimensions as follows:
% Pixels along the 'X' direction
% Pixels along the 'Y' direction
% RGB (0-255)
% Image number

close all; clc;
load("train_32x32.mat");
fprintf("Displaying 100 random images taken from the dataset being used for training the nerual network.\n")
displayTrainingImages(X);
fprintf("Press any button to train the neural network")
pause
