#include "cuSIFT.h"

SiftData::SiftData(int maxPts, bool host, bool dev, int numDevices) {

  // Initialize CUDA devices
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  numDevices = std::min(nDevices - 1, numDevices);
  deviceInit(numDevices);

#ifdef VERBOSE
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, numDevices);
  printf("Device Number: %d\n", numDevices);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
    2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
#endif

  // Initialize SiftData object
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