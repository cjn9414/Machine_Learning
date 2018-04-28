function accuracy = predict(X, y, rolled_weights, hidden_layer_size, input_layer_size, output_layer_size)
  
  m = size(y , 1); % Number of testing examples
  Theta1 = reshape(rolled_weights(1:(hidden_layer_size*(input_layer_size+1))), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(rolled_weights(1+(hidden_layer_size*(input_layer_size+1)):end), output_layer_size, hidden_layer_size + 1);
  a1 = [ones(size(X,1), 1) X];
  z2 = Theta1*a1';
  a2 = sigmoid(z2);
  a2 = [ones(1, size(a2, 2)); a2];
  z3 = Theta2*a2;
  h = sigmoid(z3);
  y_vector = zeros(m, 1);
  for (i = 1:m)
    row = y(i, :);
    [~, y_vector(i)] = find(row);
  end
  [prediction index] = max(h, [], 1);
  accuracy = sum((index' == y_vector))/m;
  
  end