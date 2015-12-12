#ifndef MATCHING_H
#define MATCHING_H

#include "cuda_runtime_api.h"
#include "cudaSift.h"
#include "cudautils.h"

double MatchSiftData(SiftData &data1, SiftData &data2);

#endif