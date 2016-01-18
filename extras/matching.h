#ifndef MATCHING_H
#define MATCHING_H

#include "cutils.h"
#include "cuSIFT.h"

typedef enum {
  MatchSiftDistanceDotProduct,
  MatchSiftDistanceL2
} MatchSiftDistance;

typedef struct {
  uint id;            // image id for tracking
  uint idx;           // SiftPoint index in SiftData.*_data
  float xim_coord;    // x coordinate in image
  float yim_coord;    // y cooordinate in image
  float x3D_coord;    // x coordinate in space
  float y3D_coord;    // y coordinate in space
  float z3D_coord;    // z coordinate in space
} SiftMatchPointDesc;

typedef struct {
  SiftMatchPointDesc point1;
  SiftMatchPointDesc point2;
  float score;
  float ambiguity;
  float error;
} SiftMatch;

// Perform an exhaustive search between all sift key points between two images
double MatchSiftData(SiftData &data1, SiftData &data2, MatchSiftDistance distance = MatchSiftDistanceL2);

#endif