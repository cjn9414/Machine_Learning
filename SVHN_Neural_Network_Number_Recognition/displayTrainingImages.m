function displayTrainingImages(X)
  
  len = size(X, 1);
  n_images_per_side = 10;
  imageLength = sqrt(size(X,2));
  X_rolled = reshape(X, len, imageLength, imageLength);
  displayMatrix = zeros(imageLength*n_images_per_side, imageLength*n_images_per_side);
  imageSelect = randi([1,len], n_images_per_side^2, 1);
  for count = 1:n_images_per_side*n_images_per_side
    x_range = 1+rem(imageLength*(count-1), n_images_per_side*imageLength);
    y_range = 1+imageLength*floor((count-1)/(n_images_per_side));
    displayMatrix(x_range:x_range+imageLength-1, y_range:y_range+imageLength-1) = X_rolled(count, :, :);
  end
  imagesc(displayMatrix')
  colormap(gray)
  end