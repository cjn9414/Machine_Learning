function grad_approx = gradientCheck(Cost, Theta)
  % Calculate the rate of change of the cost function with respect to the change of theta
  % epsilon will represent very small cahnges in theta to calculate approximate gradient
  
  % Initialize values
  epsilon = 1e-4;
  epsilon_vector = zeros(size(Theta));
  grad_approx = zeros(size(Theta));
  
  % Calculate gradient for each element in theta matrix individually
  for iter = 1:numel(Theta)
    epsilon_vector(iter) = epsilon;
    upperTheta = Cost(Theta+epsilon_vector);
    lowerTheta = Cost(Theta-epsilon_vector);
    grad_approx(iter) = (upperTheta-lowerTheta)/(2*epsilon);
    epsilon_vector(iter) = 0;
  end