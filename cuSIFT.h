#ifndef CUSIFT_H
#define CUSIFT_H

#include <iostream>
#include "cuImage.h"
#include "cutils.h"

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
  SiftData(int maxPts = 1024, bool host = false, bool dev = false, int numDevices = 0);
  ~SiftData();
};

void InitCuda(int devNum = 0);
void ExtractSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
void ExtractRootSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);

#endif
