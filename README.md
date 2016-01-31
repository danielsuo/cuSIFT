### README
This library is a CUDA implementation for various [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) (Scale Invariant Feature Transform) operations and a few helper functions such as computing homographies and estimating rigid transforms using RANSAC. We borrow from Mårten Björkman's CudaSift library.

Extra functionality lives, unsurprisingly, in the ```extras``` directory.

### Usage
This package depends on CMake for compilation and OpenCV for image containers. See ```main.cpp``` for example usage.

### Changelog
- 2015-01-14 v0.2.0: Feature updates
  - Add ExtractRootSift function based on [this](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) paper
  - Add L2 distance option in SIFT matching
  - Add rigid transform on 2D
- 2015-12-13 v0.1.0: Initial release
  - Add estimating rigid transforms via RANSAC
  - Refactor code not related to SIFT computation into extras (e.g., sift matching, finding homographies, finding rigid transforms)
