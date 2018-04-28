function debugBackprop(lambda)
  
  % In order to confirm that my backpropogation algorithm is running 
  % correctly, I am going to make a very small Neural Network that 
  % will compare my backpropogation algorithm to a gradient calculating
  % algorithm. These two algorithms should output very similar data.
  
  input_layer_size = 5;
  hidden_layer_size = 7;
  output_layer_size = 5;
  m = 10; % Number of training examples
  
  Theta1 = zeros(hidden_layer_size, input_layer_size + 1);
  Theta2 = zeros(output_layer_size, hidden_layer_size + 1);
  X = zeros(input_layer_size, m)';
  y = eye(output_layer_size)(1 + mod(1:m, output_layer_size), :);
  
  [Theta1 X] = initializeWeights(Theta1, X);
  [Theta1 Theta2] = initializeWeights(Theta1, Theta2);
  theta_unrolled = [Theta1(:); Theta2(:)];
  
  costFunc = @(t) costFunction(X, t, y, lambda, hidden_layer_size, input_layer_size, output_layer_size);
  [backprop_cost backprop_grad] = costFunc(theta_unrolled);
  grad_approx = gradientCheck(costFunc, theta_unrolled);
  fprintf("If backpropogation is working correctly, the following values should be very similar\n")
  disp([grad_approx(1:20) backprop_grad(1:20)]);
  diff = norm(grad_approx-backprop_grad)/norm(grad_approx+backprop_grad)
  end