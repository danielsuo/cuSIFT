run('../build/vlfeat/toolbox/vl_setup');

im = imread('~/Dropbox/cuSIFT/test/data/gray1.jpg');
% [f, d] = vl_sift(im2single(im), 'Octaves', floor(log2(min(size(im)))), 'Verbose');
[f, d] = vl_sift(single(im), 'Octaves', floor(log2(min(size(im)))), 'Verbose');
im2 = reshape(fread(fopen('./data/gray1', 'rb'), 640 * 480, 'single'), 640, 480)';
[f, d] = vl_sift(single(im2), 'Octaves', floor(log2(min(size(im)))), 'Verbose');

fout = fopen('./data/sift1', 'rb');
data = reshape(fread(fout, 884 * 4, 'single'), 4, 884);
fclose(fout);

% Add one to X and Y to convert to MATLAB indexing
data(1, :) = data(1, :) + 1;
data(2, :) = data(2, :) + 1;


imshow(im);
h1 = vl_plotframe(f);
h2 = vl_plotframe(data);
set(h1, 'color', 'b', 'linewidth', 2);
set(h2, 'color', 'y', 'linewidth', 2, 'linestyle', '--');