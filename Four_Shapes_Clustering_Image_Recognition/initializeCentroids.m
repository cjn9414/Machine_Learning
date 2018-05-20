function centroids = initializeCentroids(K, data)
  m = size(data, 1);
  idx = randperm(m);
  for iter = 1:K
    centroids(iter, :) = data(idx(iter), :);
  end
end
