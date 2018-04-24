function displayTrainingImages(X)
  
  len = size(X, 4);
  n_images_per_side = 10;
  imageLength = size(X,1);
  displayMatrix = zeros(imageLength*n_images_per_side, imageLength*n_images_per_side, 3);
  imageSelect = randi([1,size(X,4)], n_images_per_side^2, 1);
  for count = 1:n_images_per_side*n_images_per_side
    x_range = 1+rem(imageLength*(count-1), n_images_per_side*size(X,1));
    y_range = 1+imageLength*floor((count-1)/(n_images_per_side));
    displayMatrix(x_range:x_range+imageLength-1, y_range:y_range+imageLength-1, :) = (1/256)*cast(X(:, :, :, imageSelect(count)), 'double');
  end
  image(displayMatrix(:, :, :))
  end