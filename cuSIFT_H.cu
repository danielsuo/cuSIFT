//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include "cutils.h"

#include "cuImage.h"
#include "cuSIFT.h"
#include "cuSIFT_D.h"
#include "cuSIFT_H.h"

#include "cuSIFT_D.cu"


void SynchronizeSift(SiftData &siftData)
{
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
#endif
}

void ExtractSiftHelper(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling)
{
  int totPts = 0;
  safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

  const int nd = NUM_SCALES + 3;
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  int size = 0;         // image sizes
  int sizeTmp = nd*h*p; // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  float *memorySub = memoryTmp + sizeTmp;

  ExtractSiftLoop(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling, memoryTmp, memorySub);
  safeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
  siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  safeCall(cudaFree(memoryTmp));
}

// void ExtractSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling)
// {
//   TimerGPU timer(0);
//   ExtractSiftHelper(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling);
//   SynchronizeSift(siftData);
//   double totTime = timer.read();

// #ifndef VERBOSE
//   printf("Total time incl memory =      %.2f ms\n", totTime);
// #endif
// }

void ExtractRootSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling)
{
  TimerGPU timer(0);
  ExtractSiftHelper(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling);
  ConvertSiftToRootSift(siftData);
  SynchronizeSift(siftData);
  double totTime = timer.read();

#ifndef VERBOSE
  printf("Total time incl memory =      %.2f ms\n", totTime);
#endif
}

extern double DynamicMain(cuImage &img, SiftData &siftData, int numOctaves, double initBlur, float thresh, float lowestScale, float edgeLimit, float *memoryTmp);

void ExtractSiftLoop(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub) 
{
  TimerGPU timer(0);
#if 1
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    cuImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub); 
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
  }
  if (lowestScale<subsampling*2.0f) 
    ExtractSiftOctave(siftData, img, initBlur, thresh, lowestScale, subsampling, memoryTmp);
#else
  DynamicMain(img, siftData, numOctaves, initBlur, thresh, lowestScale, 10.0f, memoryTmp);
#endif
  double totTime = timer.read();
#ifdef VERBOSE
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
}

void ExtractSiftOctave(SiftData &siftData, cuImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nd = NUM_SCALES + 3;
  TimerGPU timer0;
  cuImage diffImg[nd];
  int w = img.width; 
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  TimerGPU timer1;
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, diffImg, baseBlur, diffScale, initBlur);
  int fstPts = 0;
  safeCall(cudaMemcpyFromSymbol(&fstPts, d_PointCounter, sizeof(int)));
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling);
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
  int totPts = 0;
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>fstPts) {
    ComputeOrientations(texObj, siftData, fstPts, totPts); 
    safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
    totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
    ExtractSiftDescriptors(texObj, siftData, fstPts, totPts, subsampling); 
  }
  safeCall(cudaDestroyTextureObject(texObj));
  double gpuTimeSift = timer4.read();

  double totTime = timer0.read();
#ifdef VERBOSE
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0) 
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts); 
#endif
}

// void InitSiftData(SiftData &data, int num, bool host, bool dev)
// {
//   data.numPts = 0;
//   data.maxPts = num;
//   int sz = sizeof(SiftPoint)*num;
// #ifdef MANAGEDMEM
//   safeCall(cudaMallocManaged((void **)&data.m_data, sz));
// #else
//   data.h_data = NULL;
//   if (host)
//     data.h_data = (SiftPoint *)malloc(sz);
//   data.d_data = NULL;
//   if (dev)
//     safeCall(cudaMalloc((void **)&data.d_data, sz));
// #endif
// }

// void FreeSiftData(SiftData &data)
// {
// #ifdef MANAGEDMEM
//   safeCall(cudaFree(data.m_data));
// #else
//   if (data.d_data!=NULL)
//     safeCall(cudaFree(data.d_data));
//   data.d_data = NULL;
//   if (data.h_data!=NULL)
//     free(data.h_data);
//   data.h_data = NULL;
// #endif
//   data.numPts = 0;
//   data.maxPts = 0;
// }

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(cuImage &res, cuImage &src, float variance)
{
  if (res.d_data == nullptr || src.d_data == nullptr) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;  
  safeCall(cudaMemcpyToSymbol(d_Kernel1, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts)
{
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);
#ifdef MANAGEDMEM
  ComputeOrientations<<<blocks, threads>>>(texObj, siftData.m_data, fstPts);
#else
  ComputeOrientations<<<blocks, threads>>>(texObj, siftData.d_data, fstPts);
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling)
{
  dim3 blocks(totPts - fstPts); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptors<<<blocks, threads>>>(texObj, siftData.m_data, fstPts, subsampling);
#else
  ExtractSiftDescriptors<<<blocks, threads>>>(texObj, siftData.d_data, fstPts, subsampling);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}


// TODO: Really, we should reimplement the end of ExtractSiftDescriptors in
// cudaSiftD.cu so we don't do L2 normalization and then L1 normalization in
// the case of RootSift
double ConvertSiftToRootSift(SiftData &siftData) {
  // For now, do naive parallelization. We are essentially creating a for loop
  // over all the sift points
  dim3 blocks(iDivUp(siftData.numPts, 16));
  dim3 threads(16);
#ifdef MANAGEDMEM
  ConvertSiftToRootSift<<<blocks, threads>>>(siftData.m_data, siftData.numPts);
#else
  ConvertSiftToRootSift<<<blocks, threads>>>(siftData.d_data, siftData.numPts);
#endif
  checkMsg("ConvertSiftToRootSift() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

double LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur)
{
  float kernel[12*16];
  float scale = baseBlur;
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) {
      kernel[16*i+j+LAPLACE_R] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += kernel[16*i+j+LAPLACE_R]; 
    }
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) 
      kernel[16*i+j+LAPLACE_R] /= kernelSum;  
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(d_Kernel2, kernel, 12*16*sizeof(float)));
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
  dim3 blocks(iDivUp(width+2*LAPLACE_R, LAPLACE_W), height);
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S); 
  LaplaceMulti<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height);
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0; 
}

double FindPointsMulti(cuImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
  float threshs[2] = { thresh, -thresh };
  float scales[NUM_SCALES];  
  float diffScale = pow(2.0f, factor);
  for (int i=0;i<NUM_SCALES;i++) {
    scales[i] = scale;
    scale *= diffScale;
  }
  safeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_EdgeLimit, &edgeLimit, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Scales, scales, sizeof(float)*NUM_SCALES));
  safeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, NUM_SCALES, subsampling); 
#else
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, NUM_SCALES, subsampling); 
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

