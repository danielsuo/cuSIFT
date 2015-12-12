#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include "cuda_runtime_api.h"
#include "cudaSift.h"
#include "cudautils.h"
#include <opencv2/core/core.hpp>

double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif