function J = computeCost(centroids, C, X)
  
  J = 0;
  m = size(X, 1)
  centroid = centroids(C, :);
  for i = 1:m
    J += sum((X(i, :) - (centroid(i, :))).^2)
  end
end
