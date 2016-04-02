// TODO: pull out / parameterize magic numbers; maybe cuSIFTOptions?
// TODO: pull out parameters into SiftData so we don't have ridiculously long function signatures
// TODO: compare ScaleDown functions vlfeat
// TODO: add cuImage member variable into SiftData?
// TODO: rename SiftData to cuSIFT
// TODO: double check peak thresh
// TODO: iliagnup 128?

#include "cuSIFT.h"
#include "cuSIFT_D.h"
#include "cuSIFT_D.cu"

SiftData::SiftData(int maxPts, bool host, bool dev) {
  this->numPts = 0;
  this->maxPts = maxPts;

  int numBytes = sizeof(SiftPoint) * maxPts;

#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&m_data, numBytes));
#else
  h_data = NULL;
  if (host) {
    h_data = (SiftPoint *)malloc(numBytes);
  }

  d_data = NULL;
  if (dev) {
    safeCall(cudaMalloc((void **)&d_data, numBytes));
  }
#endif
}

SiftData::~SiftData() {
#ifdef MANAGEDMEM
  safeCall(cudaFree(m_data));
#else
  if (d_data != NULL) {
    safeCall(cudaFree(d_data));
  }
  d_data = NULL;

  if (h_data != NULL) {
    free(h_data);
  }
  h_data = NULL;
#endif
  numPts = 0;
  maxPts = 0;
}

void SiftData::Synchronize() {
#ifdef MANAGEDMEM
  safeCall(cudaDeviceSynchronize());
#else
  if (h_data)
    safeCall(cudaMemcpy(h_data, d_data, sizeof(SiftPoint) * numPts, cudaMemcpyDeviceToHost));
#endif
}

void SiftData::Extract(float *im, int width, int height, float subsampling) {
  cuImage *cuIm = new cuImage(width, height, im);
  
  TimerGPU timer(0);

  int totPts = 0;

  // Keep track of total number of sift points as well as the maximum number
  safeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &this->maxPts, sizeof(int)));

  // TODO: what is this? why plus 3? This is for temporary space
  // TODO: move NUM_SCALES over
  const int nd = NUM_SCALES + 3;

  // Grab width, height, pitch
  int w = cuIm->width;
  int h = cuIm->height;
  int p = cuIm->pitch;

  int size = 0;             // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes

  // Determine how much memory to allocate for extraction
  for (int i = 0; i < numOctaves; i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;

  // TODO: ?? Size = size of height * pitch for each octave + (scales + 3) * height * pitch + ...
  // Return pitch CUDA allocates
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));

  // TODO: memorySub vs memoryTmp?
  float *memorySub = memoryTmp + sizeTmp;

  ExtractSiftLoop(*cuIm, numOctaves, initBlur, subsampling, memoryTmp, memorySub);
  // ExtractSiftLoop2(cuIm.get(), memoryTmp, memorySub);

  // Copy back number of points found
  safeCall(cudaMemcpyFromSymbol(&this->numPts, d_PointCounter, sizeof(int)));

  // We should only keep up to maxPts number of SiftPoints
  this->numPts = (this->numPts < this->maxPts ? this->numPts : this->maxPts);
  safeCall(cudaFree(memoryTmp));

  Synchronize();
  double totTime = timer.read();
  delete cuIm;

#ifndef VERBOSE
  printf("Total time incl memory =      %.2f ms\n", totTime);
#endif
}

// TODO: bring rootsift back
// void ExtractRootSift(SiftData &siftData, cuImage &img, int numOctaves, double initBlur, float subsampling)
// {
//   TimerGPU timer(0);
//   ExtractSiftHelper(siftData, img, numOctaves, initBlur, thresh, lowestScale, subsampling);
//   ConvertSiftToRootSift(siftData);
//   SynchronizeSift(siftData);
//   double totTime = timer.read();

// #ifndef VERBOSE
//   printf("Total time incl memory =      %.2f ms\n", totTime);
// #endif
// }

// void SiftData::ExtractSiftLoop2(cuImage *img, float *memoryTmp, float *memorySub) {
//   TimerGPU timer(0);

//   float currBlur = initBlur;
//   float currSubsampling = initSubsampling;

//   for (int octaveIndex = 0; octaveIndex < numOctaves; octaveIndex++) {
//     fprintf(stderr, "Processing octave %d\n", octaveIndex);
//     if (lowestScale < currSubsampling * 2.0f) {
//       ExtractSiftOctave(*img, currBlur, peakThresh, lowestScale, currSubsampling, memoryTmp);
//       if (octaveIndex > 0) {
//         delete img;
//       }
//     }

//     int w = img->width / 2;
//     int h = img->height / 2;
//     int p = iAlignUp(w, 128);

//     // TODO: what happens if we have odd w or h?
//     fprintf(stderr, "Making image\n");
//     cuImage *subImg = new cuImage(w, h, memorySub, false);
//     fprintf(stderr, "Scaling down image\n");
//     ScaleDown(*subImg, *img, 0.5f);

//     currBlur = (float)sqrt(currBlur * currBlur + 0.5f * 0.5f) / 2.0f;
//     currSubsampling *= 2.0f;
//     memorySub += h / 2 * p;

//     img = subImg;
//   }

//   double totTime = timer.read();
// #ifdef VERBOSE
//   printf("ExtractSift time total =      %.2f ms\n\n", totTime);
// #endif
// }

// TODO: subsampling? lowest scale?
void SiftData::ExtractSiftLoop(cuImage &img, int numOctaves, double initBlur, float subsampling, float *memoryTmp, float *memorySub) 
{
  TimerGPU timer(0);

  int w = img.width;
  int h = img.height;
  if (numOctaves > 1) {
    cuImage *subImg = new cuImage(w / 2, h / 2, memorySub);
    int p = iAlignUp(w / 2, 128);
    
    ScaleDown(*subImg, img, 0.5f);

    // TODO: Why alls this magicness
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;

    ExtractSiftLoop(*subImg, numOctaves - 1, totInitBlur, subsampling * 2.0f, memoryTmp, memorySub + (h / 2) * p);
    delete subImg;
  }

  if (lowestScale<subsampling * 2.0f) {
    ExtractSiftOctave(img, initBlur, subsampling, memoryTmp);
  }

  double totTime = timer.read();
#ifdef VERBOSE
  printf("ExtractSift time total =      %.2f ms\n\n", totTime);
#endif
}

void SiftData::ExtractSiftOctave(cuImage &img, double initBlur, float subsampling, float *memoryTmp)
{
  // TODO: again, what is this?
  const int nd = NUM_SCALES + 3;
  TimerGPU timer0;
  cuImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = img.pitch;
  for (int i = 0; i < nd - 1; i++) {
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i * p * h);
  }

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

  // Pull out thresholds
  FindPointsMulti(diffImg, sigma, 1.0f/NUM_SCALES, subsampling);
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
  int totPts = 0;
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts < this->maxPts ? totPts : this->maxPts);
  if (totPts>fstPts) {
    ComputeOrientations(texObj, fstPts, totPts); 
    safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
    totPts = (totPts < this->maxPts ? totPts : this->maxPts);
    ExtractSiftDescriptors(texObj, fstPts, totPts, subsampling); 
  }
  safeCall(cudaDestroyTextureObject(texObj));
  double gpuTimeSift = timer4.read();

  double totTime = timer0.read();
#ifdef VERBOSE
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts < this->maxPts ? totPts : this->maxPts);
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

// General strategy outlined here: http://docs.nvidia.com/cuda/samples/3_Imagi
// ng/convolutionSeparable/doc/convolutionSeparable.pdf
// TODO: convert to cuImage member function?
// TODO: investigate SCALEDOWN_W warps (160 x 16) -> chosen for apron size
double ScaleDown(cuImage &res, cuImage &src, float variance) {
  // Make sure we have allocated device data for both source and resource
  if (res.d_data == NULL || src.d_data == NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }

  // 5-pixel linear gaussian kernel
  float h_Kernel[5];

  // Normalizing factor
  float kernelSum = 0.0f;

  // Compute kernel values. We only access n / 2 + 1 of the values in the CUDA
  // kernel, but we compute all of the results so that we can normalize. We
  // can do without computing the extra exp here as well, but maybe this was
  // just easier?
  for (int j = 0; j < 5; j++) {
    h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
    kernelSum += h_Kernel[j];
  }

  // Normalize kernel values by kernelSum
  for (int j = 0; j < 5; j++) {
    h_Kernel[j] /= kernelSum;  
  }

  // Pass kernel to device
  safeCall(cudaMemcpyToSymbol(d_Kernel1, h_Kernel, 5 * sizeof(float)));

  // Set number of blocks and threads; chosen for apron size (i.e., because we
  // are doing convolution, we need to access memory outside a block; to make
  // this efficient, we want to bring in data outside the block)
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);

  // Start CUDA kernel
  ScaleDown_D<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double SiftData::ComputeOrientations(cudaTextureObject_t texObj, int fstPts, int totPts) {
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);
#ifdef MANAGEDMEM
  ComputeOrientations_D<<<blocks, threads>>>(texObj, m_data, fstPts);
#else
  ComputeOrientations_D<<<blocks, threads>>>(texObj, d_data, fstPts);
#endif
  checkMsg("ComputeOrientations_D() execution failed\n");
  return 0.0;
}

double SiftData::ExtractSiftDescriptors(cudaTextureObject_t texObj, int fstPts, int totPts, float subsampling) {
  dim3 blocks(totPts - fstPts); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptors_D<<<blocks, threads>>>(texObj, m_data, fstPts, subsampling);
#else
  ExtractSiftDescriptors_D<<<blocks, threads>>>(texObj, d_data, fstPts, subsampling);
#endif
  checkMsg("ExtractSiftDescriptors_D() execution failed\n");
  return 0.0; 
}


// TODO: Really, we should reimplement the end of ExtractSiftDescriptors in
// cudaSiftD.cu so we don't do L2 normalization and then L1 normalization in
// the case of RootSift
double SiftData::ConvertSiftToRootSift() {
  // For now, do naive parallelization. We are essentially creating a for loop
  // over all the sift points
  dim3 blocks(iDivUp(numPts, 16));
  dim3 threads(16);
#ifdef MANAGEDMEM
  ConvertSiftToRootSift_D<<<blocks, threads>>>(m_data, numPts);
#else
  ConvertSiftToRootSift_D<<<blocks, threads>>>(d_data, numPts);
#endif
  checkMsg("ConvertSiftToRootSift_D() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

double SiftData::LaplaceMulti(cudaTextureObject_t texObj, cuImage *results, float baseBlur, float diffScale, float initBlur) {
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
  LaplaceMulti_D<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height);
  checkMsg("LaplaceMulti_D() execution failed\n");
  return 0.0; 
}

double SiftData::FindPointsMulti(cuImage *sources, float scale, float factor, float subsampling) {
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
  float threshs[2] = { peakThresh, -peakThresh };
  float scales[NUM_SCALES];  
  float diffScale = pow(2.0f, factor);
  for (int i=0;i<NUM_SCALES;i++) {
    scales[i] = scale;
    scale *= diffScale;
  }

  // TODO: rename d_Threshold, d_EdgeLimit?
  safeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_EdgeLimit, &edgeThresh, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Scales, scales, sizeof(float)*NUM_SCALES));
  safeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti_D<<<blocks, threads>>>(sources->d_data, m_data, w, p, h, NUM_SCALES, subsampling); 
#else
  FindPointsMulti_D<<<blocks, threads>>>(sources->d_data, d_data, w, p, h, NUM_SCALES, subsampling); 
#endif
  checkMsg("FindPointsMulti_D() execution failed\n");
  return 0.0;
}
