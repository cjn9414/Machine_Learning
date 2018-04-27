function [J gradient] = costFunction(X_unrolled, rolled_weights, y, lambda, hidden_layer_size, input_layer_size, output_layer_size)
  %Three layers, two theta values
  m = size(y, 1); % Number of training examples
  Theta1 = reshape(rolled_weights(1:(hidden_layer_size*(input_layer_size+1))), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(rolled_weights(1+hidden_layer_size*(input_layer_size+1):end), output_layer_size, hidden_layer_size + 1);
  % Calculating hypothesis
  a1 = double(X_unrolled)'; % Creating m-rows, and n-cols, where n is the number of parameters.  
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2, 1), 1) a2];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  h = a3;
  % Calculating cost
  y_logical = eye(output_layer_size)(y,:);
  
  J = -(1/m)*sum(sum(log(h).*y_logical + log(1-h).*(1-y_logical)));
  J += (lambda/(2*m))*sum(sum(Theta1(:, 2:end).^2)); %Regularization
  J += (lambda/(2*m))*sum(sum(Theta2(:, 2:end).^2)); %Regularization
  
  % Backpropogation 
  D3 = h - y_logical;
  D2 = (D3*Theta2(:, 2:end)).*sigmoidGradient(z2);
  Delta1 = a1'*D2;
  Delta2 = a2'*D3;
  
  % Thetas has already served their purpose, can now modify to calculate gradient
  
  Theta1 = [zeros(size(Theta1, 1), 1) Theta1(1:end, 2:end)]; % Regularization
  Theta2 = [zeros(size(Theta2, 1), 1) Theta2(1:end, 2:end)]; % Regularization
  Theta1_gradient = (Delta1/m)' + (lambda/m)*Theta1;
  Theta2_gradient = (Delta2/m)' + (lambda/m)*Theta2;
  gradient = [Theta1_gradient(:); Theta2_gradient(:)];
end