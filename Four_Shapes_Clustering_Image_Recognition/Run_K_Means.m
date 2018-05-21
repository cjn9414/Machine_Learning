function [centroid_set, C] = Run_K_Means(train_data, centroid_set)
  for iter = 1:100
    m = size(train_data, 1);
    K = size(centroid_set, 1);
    C = zeros(m);
    for j = 1:K
      dist(j, :) = sum((bsxfun(@minus, train_data, centroid_set(j, :))).^2, 2);
    end
    [~, C] = min(dist);
    for i = 1:K
      cent_ex = find(C == i);
      if (numel(cent_ex) > 0)
        centroid_set(i, :) = mean(train_data(cent_ex, :));
      end
    end
  end
end