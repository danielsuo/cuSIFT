#ifndef RIGIDTRANSFORM_H
#define RIGIDTRANSFORM_H

#include "cuda_runtime_api.h"
#include <opencv2/core/core.hpp>
#include "cudautils.h"

void FindRigidTransform(const float *h_coord, int *h_randPts, float *Rt_relative, int *numInliers,
                        int numLoops, int numPts, float thresh2);

// Convenience function to take cv::Mat data and generate random indices
void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
                            float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh);

#endif