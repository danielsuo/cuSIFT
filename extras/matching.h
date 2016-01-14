#ifndef MATCHING_H
#define MATCHING_H

#include "cudautils.h"
#include "cudaSift.h"

typedef enum {
  MatchSiftDistanceDotProduct,
  MatchSiftDistanceL2
} MatchSiftDistance;

// Perform an exhaustive search between all sift key points between two images
double MatchSiftData(SiftData &data1, SiftData &data2, MatchSiftDistance distance = MatchSiftDistanceL2);

#endif