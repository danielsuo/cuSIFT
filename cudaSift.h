#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"

typedef struct {
  float xpos;
  float ypos;   
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
  int valid;
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;    // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

void InitCuda(int devNum = 0);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
void ExtractRootSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);

#endif
