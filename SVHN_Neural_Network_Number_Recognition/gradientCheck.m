function grad_approx = gradientCheck(Cost, Theta)
  
  epsilon = 1e-4;
  epsilon_vector = zeros(size(Theta));
  grad_approx = zeros(size(Theta));
  for iter = 1:numel(Theta)
    epsilon_vector(iter) = epsilon;
    upperTheta = Cost(Theta+epsilon_vector);
    lowerTheta = Cost(Theta-epsilon_vector);
    grad_approx(iter) = (upperTheta-lowerTheta)/(2*epsilon);
    epsilon_vector(iter) = 0;
  end