#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>

#ifdef WIN32
#include <intrin.h>
#endif

#include "cuda_runtime_api.h"

inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
inline int iDivDown(int a, int b) { return a / b; }
inline int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }
inline int iAlignDown(int a, int b) { return a - a % b; }

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __safeThreadSync(const char *file, const int line)
{
  cudaError err = cudaThreadSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline bool deviceInit(int dev)
{
  int deviceCount;
  safeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    return false;
  }
  if (dev < 0) dev = 0;           
  if (dev > deviceCount-1) dev = deviceCount - 1;
  cudaDeviceProp deviceProp;
  safeCall(cudaGetDeviceProperties(&deviceProp, dev));
  if (deviceProp.major < 1) {
    fprintf(stderr, "error: device does not support CUDA.\n");
    return false;         
  }
  safeCall(cudaSetDevice(dev));
  return true;
}

// Device initialization convenience function
inline void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices - 1, devNum);
  deviceInit(devNum);

#ifdef VERBOSE
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
    2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
#endif
}

class TimerGPU {
public:
  cudaEvent_t start, stop; 
  cudaStream_t stream;
  TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, stream); 
  }
  ~TimerGPU() {
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);  
  }
  float read() {
    cudaEventRecord(stop, stream); 
    cudaEventSynchronize(stop); 
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};

class TimerCPU
{
  static const int bits = 10;
public:
  long long beg_clock;
  float freq;
  TimerCPU(float freq_) : freq(freq_) {   // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits) {
#ifdef WIN32
    return __rdtsc()/(1LL<<bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
    return ((long long)high<<(32-bits)) | ((long long)low>>bits);
#endif
  }
  float read() {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1<<bits)*Kcycles/freq/1e3f;
    return time;
  }
};

#endif

