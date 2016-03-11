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

  // maxPts: allocate memory for up to maxPts SiftPoints
  // host: allocate memory on host?
  // dev: allocate memory on device?
  // numDevices: number of devices to use
  SiftData(int maxPts = 1024, bool host = false, bool dev = false);
  ~SiftData();

  void Synchronize();

  // Extract from float *
  void Extract(float *im, int width, int height, int numOctaves, double initBlur,
    float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
  void ExtractSiftLoop(cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
  void ExtractSiftOctave(cuImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);

  double ComputeOrientations(cudaTextureObject_t texObj, int fstPts, int totPts);
  double ExtractSiftDescriptors(cudaTextureObject_t texObj, int fstPts, int totPts, float subsampling);
  double LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur);
  double FindPointsMulti(cuImage *sources, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);

  double ConvertSiftToRootSift();

private:
  // ExtractFromCuImage();

  // // To get rid of
  // ExtractHelper();
  // ExtractSiftLoop();
  // ExtractSiftOctave();

  // static SiftData *ExtractSift(float *image, int numOctaves = 6, double initBlur = 0.0f, 
  //   float thresh = 0.1f, float lowestScale = 0.0f, float subsampling = 1.0f);
};

// void ExtractSiftLoop(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
// void ExtractSiftOctave(SiftData &siftData, cuImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(cuImage &res, cuImage &src, float variance);

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling);
double ConvertSiftToRootSift(SiftData &siftData);
double LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur);
double FindPointsMulti(cuImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);


// class cuSIFT {
// public:
//   int width, height;
//   int pitch;
//   float *h_data;
//   float *d_data;
//   float *t_data;
//   bool d_internalAlloc;
//   bool h_internalAlloc;

//   int numPts;         // Number of available Sift points
//   int maxPts;         // Number of allocated Sift points
// #ifdef MANAGEDMEM
//   SiftPoint *m_data;    // Managed data
// #else
//   SiftPoint *h_data;  // Host (CPU) data
//   SiftPoint *d_data;  // Device (GPU) data
// #endif

//   cuSIFT(int width, int height, float *h_data, int maxPts = 1024, bool host = false, bool dev = false);
//   ~cuSIFT();

//   void AllocateWithHostMemory(int width, int height, float *h_data);
//   void Allocate(int width, int height, int pitch, bool withHost, float *d_data = nullptr, float *h_data = nullptr);
//   double DeviceToHost();
//   double HostToDevice();
// };

// void InitCuda(int devNum = 0);
// void ExtractSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
// void ExtractRootSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
// void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
// void FreeSiftData(SiftData &data);

#endif
