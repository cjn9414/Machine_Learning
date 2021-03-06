function Process_Data()

imgsize = [200 200];

triangle_dir = './triangle';
circle_dir = './circle';
star_dir = './star';
square_dir = './square';


fprintf('Loading all triangles...\n')
triangles = dir([triangle_dir, '/*.png']);

fprintf('Loading all circles...\n')
circles = dir([circle_dir, '/*.png']);

fprintf('Loading all stars...\n')
stars = dir([star_dir, '/*.png']);

fprintf('Loading all squares...\n')
squares = dir([square_dir, '/*.png']);

shapes = [triangles(1:100); circles(1:100); stars(1:100); squares(1:100)];

clear triangles circles stars squares;

fprintf("All shape data has been loaded into MATLAB. Press any button to reformat the data\n")
pause

m = (size(shapes, 1));
data = zeros(m, imgsize(1)*imgsize(2) + 1);

for i = 1:size(shapes, 1)
    fname = shapes(i).name;
    if (i <= 100)
      data(i, (imgsize(1)*imgsize(2))+1) = 1; 
      folder = 'triangle/';
    elseif (i <= 200)
      data(i, (imgsize(1)*imgsize(2))+1) = 2; 
      folder = 'circle/';
    elseif (i <= 300)
      data(i, (imgsize(1)*imgsize(2))+1) = 3; 
      folder = 'star/';
    else 
      data(i, (imgsize(1)*imgsize(2))+1) = 4; 
      folder = 'square/';
    end
    data(i, 1:imgsize(1)*imgsize(2)) = (imread([folder fname])(:))';
end

fprintf("All data has been reformatted. Press any button to write the data into a file.\n")
pause

n = size(data, 1);
idx = randperm(n);
data_shuffle = zeros(size(data));
for i = 1:n
  data_shuffle(i, :) = data(idx(i), :);
end

data = data_shuffle;
save('data.mat', 'data');

clear all;

end