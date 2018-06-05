clear all;
K = 4;

if ~exist('data.mat')
  Process_Data();
end

fprintf("Press any button to load data file.\n")
pause

load('data.mat');
n = size(data, 2) - 1;
train_data = data(1:300, 1:n);

fprintf("Data loaded into Octave. Press any button to perform dimensionality reduction.\n")
pause

%X_reduced = reduceDimensions(scaleFeatures(train_data));

fprintf("Dimensionality reduction completed. Press any button to continue.\n")
pause

test_data = data(301:end, 1:n);
labels = data(301:end, n+1);
size(labels)
pause
clear data;
m = size(train_data, 1);
n = size(train_data, 2);
numb_trials = 100;
centroids = zeros(K, n, numb_trials);
J = zeros(numb_trials);

for trial = 1:numb_trials
  centroids(:, :, trial) = initializeCentroids(K, train_data);
  [centroids(:, :, trial), C] = Run_K_Means(train_data, centroids(:, :, trial));
  J(trial) = computeCost(centroids(:, :, trial), C, X);
end

[cost idx] = min(J);

optimalCentroids = centroids(:, :, idx);
accuracy = computeAccuracy(optimalCentroids, test_data, labels);

fprintf("The accuracy of this clustering algorithm is approximately %f\n", accuracy*100);