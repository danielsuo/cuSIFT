#include "matching.h"

#define FLT_MAX 999.0

//================= Device matching functions =====================//

// sift1: array of SiftPoints from image 1
// sift2: array of SiftPoints from image 2
// corrData: sizeof(float) * numPts1 * numPts2 (rounded to the least larger multiple of 16)
// numPts1: number of SiftPoints from image 1
// numPts2: number of SiftPoints from image 2
__global__ void ComputeDistance(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2) {
  // Initialize arrays to hold 16 sift points descriptors (128 floats) at a
  // time from images 1 and 2
  __shared__ float siftPoints1[16 * 128];
  __shared__ float siftPoints2[16 * 128];

  // Get thread ids; threads are in 16x16x1 blocks
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Get pointer to sift descriptor data
  const float *ptr1 = sift1[min(numPts1 - 1, blockIdx.x * 16 + ty)].data;
  const float *ptr2 = sift2[min(numPts2 - 1, blockIdx.y * 16 + ty)].data;

  // Load data into shared memory from global memory
  for (int i = 0; i < 8; i++) {
    siftPoints1[128 * ty + 16 * i + tx] = ptr1[16 * i + tx];
    siftPoints2[128 * ty + 16 * i + tx] = ptr2[16 * i + tx];
  }

  // Synchronize threads once all the data has been loaded. Now, instead of ty
  // and tx as identifiers to organize data, we use ty and tx to indicate key
  // point indices
  __syncthreads();

  // Get descriptor data. By using ty and tx to get two different points, we
  // make sure we get all combinations of two points from the block of 16 key
  // points from each image
  const float *pt1 = &siftPoints1[ty * 128];
  const float *pt2 = &siftPoints2[tx * 128];

  // Compute similarity score
  float sum = 0.0f;
  for (int i = 0; i < 128; i++) {
    // avoid bank conflicts
    int itx = (i + tx) & 127;
    sum += pt1[itx] * pt2[itx];
  }

  // Get the global point index, not the local index within our 16x16 chunk
  const int p1 = blockIdx.x * 16 + ty;
  const int p2 = blockIdx.y * 16 + tx;

  // Make sure p1 and p2 are both within bounds
  if (p1 < numPts1)
    corrData[p1 * gridDim.y * 16 + p2] = (p2 < numPts2 ? sum : -1.0f);
}

// Compute L2 norm. Note that because sift descriptors are 128-dimensional
// unit vectors, we can compute the L2 norm as follow: L2(pt1, pt2) = ||pt1 -
// pt2||^2 = ||pt1||^2 + ||pt2||^2 - 2x'y = 2 - 2x'y
__global__ void ComputeL2Distance(float *corrData, int numPts1) {
  // Get the global point index, not the local index within our 16x16 chunk
  const int p1 = blockIdx.x * 16 + threadIdx.x;
  const int p2 = blockIdx.y * 16 + threadIdx.y;

  // Make sure p1 and p2 are both within bounds
  if (p1 < numPts1) {
    const int idx = p1 * gridDim.y * 16 + p2;
    if (corrData[idx] > -1) corrData[idx] = 2 - 2 * corrData[idx];
    else corrData[idx] = FLT_MAX;
  }
}

__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize) {
  // We are processing 16 points at a time
  __shared__ float maxScore[16 * 16];
  __shared__ float maxScore2[16 * 16];
  __shared__ int maxIndex[16 * 16];

  // Get thread indices, which both range from 0 to 15
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Range from 0 to 255
  const int idx = ty * 16 + tx;

  // Get point index
  int p1 = blockIdx.x * 16 + ty;
  p1 = (p1 >= numPts1 ? numPts1 - 1 : p1);

  // Initialize scores
  maxScore[idx] = -1.0f;
  maxScore2[idx] = -1.0f;
  maxIndex[idx] = -1;

  // Synchronize threads before beginning to find match.
  __syncthreads();

  // Get the correlation data for point p1 against all the p2s
  float *corrs = &corrData[p1 * corrWidth];

  // Loop through all of the p2s
  for (int i = tx; i < corrWidth; i += 16) {

    // Correlation between p1 and the ith p2
    float val = corrs[i];

    // Find the two highest scores for ratio testing later
    if (val > maxScore[idx]) {
      maxScore2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    } else if (val > maxScore2[idx]) {
      maxScore2[idx] = val;
    }
  }

  __syncthreads();

  for (int len = 8; len > 0; len /= 2) {
    if (tx < 8) {
      float val = maxScore[idx + len];
      int i = maxIndex[idx + len];
      if (val > maxScore[idx]) {
        maxScore2[idx] = maxScore[idx];
        maxScore[idx] = val;
        maxIndex[idx] = i;
      } else if (val > maxScore2[idx]) {
        maxScore2[idx] = val;
      }
      float val2 = maxScore2[idx + len];
      if (val2 > maxScore2[idx])
       maxScore2[idx] = val2;
   }
    __syncthreads();
  }

  if (tx == 6){}
    sift1[p1].score = maxScore[ty * 16];
  if (tx == 7)
    sift1[p1].ambiguity = maxScore2[ty * 16] / (maxScore[ty * 16] + 1e-6);
  if (tx == 8)
    sift1[p1].match = maxIndex[ty * 16];
  if (tx == 9)
    sift1[p1].match_xpos = sift2[maxIndex[ty * 16]].coords2D[0];
  if (tx == 10)
    sift1[p1].match_ypos = sift2[maxIndex[ty * 16]].coords2D[1];
  __syncthreads();
}

__global__ void FindMinCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize) {
  // We are processing 16 points at a time
  __shared__ float minScore[16 * 16];
  __shared__ float minScore2[16 * 16];
  __shared__ int minIndex[16 * 16];

  // Get thread indices, which both range from 0 to 15
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Range from 0 to 255
  const int idx = ty * 16 + tx;

  // Get point index
  int p1 = blockIdx.x * 16 + ty;
  p1 = (p1 >= numPts1 ? numPts1 - 1 : p1);

  // Initialize scores
  minScore[idx] = FLT_MAX;
  minScore2[idx] = FLT_MAX;
  minIndex[idx] = -1;

  // Synchronize threads before beginning to find match.
  __syncthreads();

  // Get the correlation data for point p1 against all the p2s
  float *corrs = &corrData[p1 * corrWidth];

  // Loop through all of the p2s
  for (int i = tx; i < corrWidth; i += 16) {

    // Correlation between p1 and the ith p2
    float val = corrs[i];

    // Find the two lowest scores for ratio testing later
    if (val < minScore[idx]) {
      minScore2[idx] = minScore[idx];
      minScore[idx] = val;
      minIndex[idx] = i;
    } else if (val < minScore2[idx]) {
      minScore2[idx] = val;
    }
  }

  __syncthreads();

  for (int len = 8; len > 0; len /= 2) {
    if (tx < 8) {
      float val = minScore[idx + len];
      int i = minIndex[idx + len];
      if (val < minScore[idx]) {
        minScore2[idx] = minScore[idx];
        minScore[idx] = val;
        minIndex[idx] = i;
      } else if (val < minScore2[idx]) {
        minScore2[idx] = val;
      }
      float val2 = minScore2[idx + len];
      if (val2 < minScore2[idx]) {
        minScore2[idx] = val2;
      }
    }
    __syncthreads();
  }

  if (tx == 6)
    sift1[p1].score = minScore[ty * 16];
  if (tx == 7)
    sift1[p1].ambiguity = minScore[ty * 16] / (minScore2[ty * 16] + 1e-6);
  if (tx == 8)
    sift1[p1].match = minIndex[ty * 16];
  if (tx == 9)
    sift1[p1].match_xpos = sift2[minIndex[ty * 16]].coords2D[0];
  if (tx == 10)
    sift1[p1].match_ypos = sift2[minIndex[ty * 16]].coords2D[1];
  __syncthreads();
}

vector<SiftMatch *> MatchSiftData(SiftData &data1,
                                  SiftData &data2,
                                  MatchSiftDistance distance,
                                  float scoreThreshold,
                                  float ambiguityThreshold,
                                  MatchType type) {
  TimerGPU timer(0);
  vector<SiftMatch *> matches;
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2) 
    return matches;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data == NULL || data2.d_data == NULL)
    return matches;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif
  
  float *d_corrData; 

  // Width is least multiple of 16 greater than numPts2
  int corrWidth = iDivUp(numPts2, 16) * 16;

  // corrSize is width by numPts1
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));

  // Create a numPts1 x numPts2 (up to least multiple of 16 greater) grid so
  // we can compare all pts in image 1 and 2. Note that half of the threads
  // will be wasted because we'll be computing the scores twice. Can we do
  // better?
  dim3 blocks(iDivUp(numPts1, 16), iDivUp(numPts2, 16));

  // each block: 16 points per block, each with 16 chunks of 8 floats for
  // 128-float descriptor
  dim3 threads(16, 16); 

  // Compute distance scores between all the sift points in image 1 against
  // all the sift points in image 2
  ComputeDistance<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
  switch(distance) {
    case MatchSiftDistanceDotProduct:
    break;

    // If we want L2 distance, we need to do one step of extra processing.
    case MatchSiftDistanceL2:
    ComputeL2Distance<<<blocks, threads>>>(d_corrData, numPts1);
    break;
  }
  
  // Synchronize threads
  safeCall(cudaThreadSynchronize());

  // Set up problem for finding matching sift. We are going to loop over all
  // the key points in the first image
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);

  switch(distance) {
    case MatchSiftDistanceDotProduct:
    FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
    break;

    // If we want L2 distance, we need to do one step of extra processing.
    case MatchSiftDistanceL2:
    FindMinCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
    break;
  }

  // Synchronize threads
  safeCall(cudaThreadSynchronize());
  checkMsg("MatchSiftPoints() execution failed\n");

  // Clean data and transfer back to host memory
  safeCall(cudaFree(d_corrData));
  if (data1.h_data != NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5 * sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }

  // TODO: When we refactor match data out of SiftPoint, move this to CUDA
  // kernel. For now, create SiftMatch vector
  for (int i = 0; i < data1.numPts; i++) {
    bool foundMatch = false;
    switch(distance) {
      case MatchSiftDistanceDotProduct:
      foundMatch = data1.h_data[i].score > scoreThreshold;
      break;
      case MatchSiftDistanceL2:
      foundMatch = data1.h_data[i].score < scoreThreshold;
      break;
    }

    foundMatch &= data1.h_data[i].ambiguity < ambiguityThreshold;

    if (foundMatch) {
      SiftMatch *match = new SiftMatch();

      // We should clear out the SiftPoint memory for score, ambiguity, but we
      // don't. They should be initialized in a constructor.
      match->pt1 = &(data1.h_data[i]);
      match->pt2 = &(data2.h_data[match->pt1->match]);
      match->score = match->pt1->score;
      match->ambiguity = match->pt1->ambiguity;

      // If we are matching in 3D, make sure we have data available
      if (type == MatchType2D || (match->pt1->coords3D[2] != 0 && match->pt2->coords3D[2] != 0)) {
        matches.push_back(match);
      } else {
        delete match;
      }
    }
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("MatchSiftData time =          %.2f ms\n", gpuTime);
#endif
  return matches;
}
