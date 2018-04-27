function [J gradient] = costFunction(X_unrolled, rolled_weights, y, lambda, hidden_layer_size, input_layer_size, output_layer_size)
  %Three layers, two theta values
  m = size(y, 1);
  Theta1 = reshape(rolled_weights(1:(hidden_layer_size*(input_layer_size+1))), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(rolled_weights(1+hidden_layer_size*(input_layer_size+1):end), output_layer_size, hidden_layer_size + 1);
  
  
  % Calculating hypothesis
  a1 = double(X_unrolled);
  %a1 = [ones(size(X_unrolled, 1), 1) a1];
  
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [ones(1, size(a2, 2)); a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  h = a3;
  
  % Calculating cost
  
  J = -(1/m)*sum(sum(log(h).*y' + log(1-h).*(1-y)'));
  %J += (lambda/(2*m))*sum(sum(Theta1.^2)); %Regularization
  %J += (lambda/(2*m))*sum(sum(Theta2.^2)); %Regularization
  J += (lambda/(2*m))*sum(sum(Theta1(:, 2:end).^2)); %Regularization
  J += (lambda/(2*m))*sum(sum(Theta2(:, 2:end).^2)); %Regularization
  
  % Backpropogation 
  y_logical = eye(size(h,1))(y,:);
  D3 = h' - y_logical;
  D2 = (D3*Theta2(:, 2:end)).*(sigmoidGradient(z2))';
  Delta1 = a1*D2;
  Delta2 = a2*D3;
  
  Theta1 = [zeros(1, size(Theta1, 2)) ; Theta1(2:end, :)];
  Theta2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
  Theta1_gradient = Delta1/m + ((lambda/m)*Theta1)';
  Theta2_gradient = Delta2/m + ((lambda/m)*Theta2)';  
  gradient = [Theta1_gradient(:); Theta2_gradient(:)];
end