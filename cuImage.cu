//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cutils.h"
#include "cuImage.h"

// Convenience function to allocate memory
void cuImage::AllocateWithHostMemory(int width, int height, float *h_data) {
  Allocate(width, height, iAlignUp(width, 128), false, nullptr, h_data);
}

// Allocate memory on the host and device
void cuImage::Allocate(int width, int height, int pitch, bool host, float *d_data, float *h_data) 
{
  // Set image width and height
  this->width = width;
  this->height = height;
  this->pitch = pitch;

  // Set device and host data
  this->d_data = d_data;
  this->h_data = h_data;
  t_data = nullptr; 

  // If pointer to device memory is nullptr, allocate
  if (this->d_data == nullptr) {
    safeCall(cudaMallocPitch((void **)&this->d_data, (size_t *)&pitch, (size_t)(sizeof(float) * width), (size_t)height));

    // Assume pitch is correctly aligned
    pitch /= sizeof(float);
    if (this->d_data == nullptr) {
      printf("Failed to allocate device data\n");
    }
    d_internalAlloc = true;
  }

  // If pointer to host memory is nullptr and we would like to allocate host memory
  if (host && this->h_data == nullptr) {
    this->h_data = (float *)malloc(sizeof(float) * pitch * height);
    h_internalAlloc = true;
  }
}

cuImage::cuImage() : 
  width(0), height(0), d_data(nullptr), h_data(nullptr), t_data(nullptr), d_internalAlloc(false), h_internalAlloc(false)
{

}

cuImage::cuImage(int width, int height, float *h_data) {
  d_internalAlloc = false;
  h_internalAlloc = false;
  AllocateWithHostMemory(width, height, h_data);
  HostToDevice();
}

cuImage::~cuImage()
{
  // If we've allocated device memory, free
  if (d_internalAlloc && d_data != nullptr) {
    safeCall(cudaFree(d_data));
  }
  d_data = nullptr;

  // If we've allocated host memory, free
  if (h_internalAlloc && h_data != nullptr) {
    free(h_data);
  }
  h_data = nullptr;

  if (t_data != nullptr) {
    safeCall(cudaFreeArray((cudaArray *)t_data));
  }
  t_data = nullptr;
}

// Copy image from host memory to device memory
double cuImage::HostToDevice()  
{
  TimerGPU timer(0);
  int p = sizeof(float) * pitch;

  // Check if both device and host data are non-null
  if (d_data != nullptr && h_data != nullptr) {
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width,
      sizeof(float) * width, height, cudaMemcpyHostToDevice));
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("HostToDevice time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

// Copy image from device memory to host memory
double cuImage::DeviceToHost()
{
  TimerGPU timer(0);
  int p = sizeof(float) * pitch;

  if (d_data != nullptr && h_data != nullptr) {
    safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p,
      sizeof(float) * width, height, cudaMemcpyDeviceToHost));
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("DeviceToHost time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}