function X_scaled = scaleFeatures(X)
  
  m = size(X, 1);
  u = sum(X)/m;
  X_scaled = X-u;
  
end
