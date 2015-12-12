#ifndef RIGIDTRANSFORM_H
#define RIGIDTRANSFORM_H

#include <opencv2/core/core.hpp>

void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
                            float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh);

#endif