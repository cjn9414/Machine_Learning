clear all;
K = 4;

if ~exist('data.mat')
  Process_Data();
end
load('data.mat');
train_data = data(1:400, :);
test_data = data(401:end, :);
m = size(train_data, 1);
n = size(train_data, 2);
numb_trials = 100;
centroids = zeros(K, n, numb_trials);
J = zeros(numb_trials);

for trial = 1:numb_trials
  centroids(:, :, trial) = initializeCentroids(K, train_data);
  %[centroids, C] = Run_K_Means(train_data, centroids); # NOT FINISHED #
  %J(trial) = computeCost(centroids, C); # NOT FINISHED #
end

[cost idx] = min(J);

optimalCentroids = centroids(:, :, idx);
%computeAccuracy(optimalCentroids, test_data); # need labeled data first #