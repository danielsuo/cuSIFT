#include "cuSIFT.h"

SiftData::SiftData(int maxPts, bool host, bool dev) {
  this->numPts = 0;
  this->maxPts = maxPts;

  int numBytes = sizeof(SiftPoint) * maxPts;

#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&m_data, numBytes));
#else
  h_data = NULL;
  if (host)
    h_data = (SiftPoint *)malloc(numBytes);
  d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&d_data, numBytes));
#endif
}

SiftData::~SiftData() {
#ifdef MANAGEDMEM
  safeCall(cudaFree(m_data));
#else
  if (d_data!=NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_data!=NULL)
    free(h_data);
  h_data = NULL;
#endif
  numPts = 0;
  maxPts = 0;
}

SiftData *ExtractSift(float *image, int numOctaves, double initBlur, float thresh, 
  float lowestScale, float subsampling) {
  SiftData *data = new SiftData(1024, true, true);

  return data;
}