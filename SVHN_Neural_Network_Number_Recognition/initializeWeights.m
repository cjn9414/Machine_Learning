function [init_Theta1 init_Theta2] = initializeWeights(Theta1, Theta2)
  epsilon = 4*sqrt(6/(size(Theta1,2)+size(Theta2,1))); % fan-in + fan-out in denominator of square root.
  init_Theta1 = rand(size(Theta1))*2*epsilon-epsilon; % range between -epsilon and +epsilon
  init_Theta2 = rand(size(Theta2))*2*epsilon-epsilon;
  end