function X_reduced = reduceDimensions(X, k)
  
  m = size(X, 1);
  n = size(X, 2)
  sigma = (1/m)*X'*X;
  [U, S, V] = svd(sigma);
  
  for k = 1:n
    if (sum(sum(S(1:k)))/sum(sum(S)) >= 0.99)
      break
    end
  end
  
  U_reduced = U(1:k);
  X_reduced = X*U_reduced;
  
end
