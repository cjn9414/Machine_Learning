function s = sigmoidGradient(z)
  
  s = sigmoid(z).*(1-sigmoid(z));
  
  end