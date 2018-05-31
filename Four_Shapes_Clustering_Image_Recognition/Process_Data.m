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

shapes = [triangles(1:1000); circles(1:1000); stars(1:1000); squares(1:1000)];

clear triangles circles stars squares;

fprintf("All shape data has been loaded into MATLAB. Press any button to reformat the data\n")
pause

const = 8; % Used to determine size of data file
m = (size(shapes, 1))/const;
data = zeros(m, imgsize(1)*imgsize(2));

for i = 1:const:size(shapes, 1)
    fname = shapes(i).name;
    if (i <= 3720)
      folder = 'triangle/';
    elseif (i <= 3720*2)
      folder = 'circle/';
    elseif (i <= (3720*2) + 3765)
      folder = 'star/';
    else 
      folder = 'square/';
    end
    data((const+i-1)/const, :) = (imread([folder fname])(:))';
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