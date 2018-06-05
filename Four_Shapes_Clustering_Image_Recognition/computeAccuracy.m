function accuracy = computeAccuracy(centroids, X, labels)

  m = size(X, 1);
  K = size(centroids, 1);
  for i = 1:K
    dist(i, :) = sum((bsxfun(@minus, X, centroids(i, :))).^2, 2);
  end

  [~, C] = min(dist);
  nCorrect = sum(C' == labels);
  accuracy = nCorrect/m;
end