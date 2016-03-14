#ifndef CUSIFT_H
#define CUSIFT_H

#include <iostream>
#include "cuImage.h"
#include "cutils.h"

using namespace std;

class SiftPoint {
public:
  float coords2D[2];
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;

  // TODO: remove these
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;

  float subsampling;
  float empty[3];
  float data[128];
  float coords3D[3];
};

class SiftData {
public:
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points

#ifdef MANAGEDMEM
  SiftPoint *m_data;    // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif

  // SIFT parameters
  int numOctaves;
  int numScales;
  double initBlur;
  float initSubsampling;
  float peakThresh;
  float edgeThresh;
  float lowestScale;

  // maxPts: allocate memory for up to maxPts SiftPoints
  // host: allocate memory on host?
  // dev: allocate memory on device?
  // numDevices: number of devices to use
  SiftData(int maxPts = 1024, bool host = false, bool dev = false);
  ~SiftData();

  void Synchronize();

  // Extract from float *
  void Extract(float *im, int width, int height, float subsampling = 1.0f);
  void ExtractSiftLoop2(cuImage *img,float *memoryTmp, float *memorySub);
  void ExtractSiftLoop(cuImage &img, int numOctaves, double initBlur, float subsampling, float *memoryTmp, float *memorySub);
  void ExtractSiftOctave(cuImage &img, double initBlur, float subsampling, float *memoryTmp);

  double ComputeOrientations(cudaTextureObject_t texObj, int fstPts, int totPts);
  double ExtractSiftDescriptors(cudaTextureObject_t texObj, int fstPts, int totPts, float subsampling);
  double LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur);
  double FindPointsMulti(cuImage *sources, float scale, float factor, float subsampling);

  double ConvertSiftToRootSift();
};

double ScaleDown(cuImage &res, cuImage &src, float variance);


#endif
