#ifndef CUSIFTH_H
#define CUSIFTH_H

#include "cutils.h"
#include "cuImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

void ExtractSiftLoop(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, cuImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(cuImage &res, cuImage &src, float variance);
double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling);
double ConvertSiftToRootSift(SiftData &siftData);
double LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur);
double FindPointsMulti(cuImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);

#endif
