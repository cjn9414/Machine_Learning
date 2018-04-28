function [J gradient] = costFunction(X, rolled_weights, y, lambda, hidden_layer_size, input_layer_size, output_layer_size)
  
  m = size(y, 1); % Number of training examples
  
  % Reform the theta matrices
  Theta1 = reshape(rolled_weights(1:(hidden_layer_size*(input_layer_size+1))), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(rolled_weights(1+(hidden_layer_size*(input_layer_size+1)):end), output_layer_size, hidden_layer_size + 1);
  
  % Calculating hypothesis
  a1 = [ones(size(X,1), 1) X];
  z2 = Theta1*a1';
  a2 = sigmoid(z2);
  a2 = [ones(1, size(a2, 2)) ; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  h = a3;
  
  % Calculating cost  
  J = -(1/m)*sum(sum(log(h)'.*y + log(1-h)'.*(1-y)));
  J += (lambda/(2*m))*sum(sum(Theta1(:, 2:end).^2)); % Regularization
  J += (lambda/(2*m))*sum(sum(Theta2(:, 2:end).^2)); % Regularization
  
  % Backpropogation 
  D3 = h' - y;
  D2 = (D3*Theta2(:, 2:end))'.*sigmoidGradient(z2);
  Delta1 = D2*a1;
  Delta2 = D3'*a2';
  
  % Thetas have already served their purpose, can now modify to calculate gradient
  
  Theta1 = [zeros(size(Theta1, 1), 1) Theta1(1:end, 2:end)]; % Regularization
  Theta2 = [zeros(size(Theta2, 1), 1) Theta2(1:end, 2:end)]; % Regularization
  Theta1_gradient = (Delta1/m) + (lambda/m)*Theta1;
  Theta2_gradient = (Delta2/m) + (lambda/m)*Theta2;
  gradient = [Theta1_gradient(:) ; Theta2_gradient(:)];
end