#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <opencv2/core/core.hpp>
#include "cutils.h"
#include "cuSIFT.h"

double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif