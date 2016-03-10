run('../build/vlfeat/toolbox/vl_setup');

% MATLAB VLFeat
im = imread('~/Dropbox/cuSIFT/test/data/gray1.jpg');
im2 = reshape(fread(fopen('./data/gray1', 'rb'), 640 * 480, 'single'), 640, 480)';
[f, d] = vl_sift(single(im2), 'Octaves', floor(log2(min(size(im)))), 'Verbose');

% C VLFeat
fout = fopen('./data/sift1', 'rb');
data = reshape(fread(fout, 884 * 4, 'single'), 4, 884);
fclose(fout);

% Add one to X and Y to convert to MATLAB indexing
data(1, :) = data(1, :) + 1;
data(2, :) = data(2, :) + 1;

% cuSIFT
fout = fopen('./data/cusift1', 'rb');
numPts = fread(fout, 1, 'uint32');
data2 = reshape(fread(fout, numPts * 4, 'single'), 4, numPts);
data2(1, :) = data2(1, :) + 1;
data2(2, :) = data2(2, :) + 1;

imshow(im);
h1 = vl_plotframe(f);
h2 = vl_plotframe(data);
h3 = vl_plotframe(data2);
set(h3, 'color', 'r', 'linewidth', 1, 'linestyle', '-.');
set(h1, 'color', 'b', 'linewidth', 2);
set(h2, 'color', 'y', 'linewidth', 2, 'linestyle', '--');
