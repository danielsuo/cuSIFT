#ifndef RIGIDTRANSFORM_H
#define RIGIDTRANSFORM_H

#include <vector>
// #include <opencv2/core/core.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include "extras/matching.h"
#include "extras/math_utils.h"
#include "cutils.h"

// Not strictly necessary, but because all other extras also include, creates
// easier interface
#include "cuSIFT.h"

typedef enum {
  RigidTransformType2D,
  RigidTransformType3D
} RigidTransformType;

// Host function that doesn't use OpenCV Mat
void EstimateRigidTransformH(const float *h_coord, float *Rt_relative, int *numInliers, int numLoops, 
                             int numPts, float thresh2, RigidTransformType type = RigidTransformType2D,
                             int *h_indices = NULL, char *h_inliers = NULL);

// Convenience function to take cv::Mat data and generate random indices
// void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
//                             float* Rt_relative, int* numInliers, 
//                             int numLoops, float thresh, RigidTransformType type = RigidTransformType2D);

// // Convenience function to take vector of SiftMatch
void EstimateRigidTransform(vector<SiftMatch *> matches, float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh, RigidTransformType type, int *h_indices = NULL, 
                             char *h_inliers = NULL);

#endif