//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cutils.h"
#include "cuImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

// Convenience function to allocate memory
void cuImage::Allocate(int w, int h, float *hostmem) {
  Allocate(w, h, iAlignUp(w, 128), false, nullptr, hostmem);
}

// Allocate memory on the host and device
void cuImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem) 
{
  // Set image width and height
  width = w;
  height = h;
  pitch = p;

  // Set device and host data
  d_data = devmem;
  h_data = hostmem;
  t_data = nullptr; 

  // if pointer to device memory is nullptr, allocate
  if (devmem == nullptr) {
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==nullptr) 
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==nullptr) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

cuImage::cuImage() : 
  width(0), height(0), d_data(nullptr), h_data(nullptr), t_data(nullptr), d_internalAlloc(false), h_internalAlloc(false)
{

}

cuImage::~cuImage()
{
  if (d_internalAlloc && d_data!=nullptr) 
    safeCall(cudaFree(d_data));
  d_data = nullptr;
  if (h_internalAlloc && h_data!=nullptr) 
    free(h_data);
  h_data = nullptr;
  if (t_data!=nullptr) 
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = nullptr;
}
  
double cuImage::Download()  
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  if (d_data!=nullptr && h_data!=nullptr) 
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double cuImage::Readback()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double cuImage::InitTexture()
{
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height)); 
  if (t_data==nullptr)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
 
double cuImage::CopyToTexture(cuImage &dst, bool host)
{
  if (dst.t_data==nullptr) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==nullptr) && (host || d_data==nullptr)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
