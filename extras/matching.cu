#include "cudaSift.h"
#include "cudautils.h"

//================= Device matching functions =====================//

__global__ void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x;
  const int p2 = blockIdx.y*16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = 16*ty + tx;
  if (ty<8)
    siftPoint[i] = ptr1[i];
  __syncthreads();
  float sum = 0.0f;
  if (p2<numPts2)
    for (int j=0;j<8;j++)
      sum += siftPoint[16*j+tx] * ptr2[16*j+tx];
  sums[i] = sum;
  __syncthreads();
  if (tx<8)
    sums[i] += sums[i+8];
  __syncthreads();
  if (tx<4)
    sums[i] += sums[i+4];
  __syncthreads();
  if (ty==0) {
    sum = sums[16*tx+0] + sums[16*tx+1] + sums[16*tx+2] + sums[16*tx+3];
    corrData[p1*gridDim.y*16 + blockIdx.y*16 + tx] = sum;
  }
  __syncthreads();
}

__global__ void MatchSiftPoints2(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoints1[16*128];
  __shared__ float siftPoints2[16*128];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const float *ptr1 = sift1[min(numPts1-1,blockIdx.x*16 + ty)].data;
  const float *ptr2 = sift2[min(numPts2-1,blockIdx.y*16 + ty)].data;
  for (int i=0;i<8;i++) {
    siftPoints1[128*ty+16*i+tx] = ptr1[16*i+tx];
    siftPoints2[128*ty+16*i+tx] = ptr2[16*i+tx];
  }
  __syncthreads();
  const int p1 = blockIdx.x*16 + ty;
  const int p2 = blockIdx.y*16 + tx;
  const float *pt1 = &siftPoints1[ty*128];
  const float *pt2 = &siftPoints2[tx*128];
  float sum = 0.0f;
  for (int i=0;i<128;i++) {
    int itx = (i + tx)&127; // avoid bank conflicts
    sum += pt1[itx]*pt2[itx];
  }
  if (p1<numPts1)
    corrData[p1*gridDim.y*16 + p2] = (p2<numPts2 ? sum : -1.0f);
}

__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize)
{
  __shared__ float maxScore[16*16];
  __shared__ float maxScor2[16*16];
  __shared__ int maxIndex[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*16 + tx;
  int p1 = blockIdx.x*16 + threadIdx.y;
  p1 = (p1>=numPts1 ? numPts1-1 : p1);
  maxScore[idx] = -1.0f;
  maxScor2[idx] = -1.0f;
  maxIndex[idx] = -1;
  __syncthreads();
  float *corrs = &corrData[p1*corrWidth];
  for (int i=tx;i<corrWidth;i+=16) {
    float val = corrs[i];
    if (val>maxScore[idx]) {
      maxScor2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    } else if (val>maxScor2[idx])
      maxScor2[idx] = val;
  }
  //if (p1==1)
  //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
  //	   tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  __syncthreads();
  for (int len=8;len>0;len/=2) {
    if (tx<8) {
      float val = maxScore[idx+len];
      int i = maxIndex[idx+len];
      if (val>maxScore[idx]) {
	maxScor2[idx] = maxScore[idx];
	maxScore[idx] = val;
	maxIndex[idx] = i;
      } else if (val>maxScor2[idx])
	maxScor2[idx] = val;
      float va2 = maxScor2[idx+len];
      if (va2>maxScor2[idx])
	maxScor2[idx] = va2;
    }
    __syncthreads();
    //if (p1==1 && tx<len) 
    //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
    //	     tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  }
  if (tx==6)
    sift1[p1].score = maxScore[ty*16];
  if (tx==7)
    sift1[p1].ambiguity = maxScor2[ty*16] / (maxScore[ty*16] + 1e-6);
  if (tx==8)
    sift1[p1].match = maxIndex[ty*16];
  if (tx==9)
    sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
  if (tx==10)
    sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  __syncthreads();
  //if (tx==0)
  //  printf("index = %d/%d, score = %.2f, ambiguity = %.2f, match = %d\n", 
  //	p1, numPts1, sift1[p1].score, sift1[p1].ambiguity, sift1[p1].match);
}

double MatchSiftData(SiftData &data1, SiftData &data2)
{
  TimerGPU timer(0);
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2) 
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif
  
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
#if 0
  dim3 blocks1(numPts1, iDivUp(numPts2, 16));
  dim3 threads1(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks1, threads1>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#else
  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#endif
  safeCall(cudaThreadSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaThreadSynchronize());
  checkMsg("MatchSiftPoints() execution failed\n");
  safeCall(cudaFree(d_corrData));
  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("MatchSiftData time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
