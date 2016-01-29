#include "rigidTransform.h"

// For 3 point only 
// X: [x0, y0, z0, x1, y1, z1, x2, y2, z2] 
// Y: [x0',y0',z0',x1,y1',z1',x2',y2',z2'] 
// xh = T * yh
// 
// d_coord: numPtsx6 matrix. First three elements per row stores the indices
// for the random points and the fourth element holds the error (i.e., output)
// d_randPts: 3xnumLoops matrix. First three elements hold indices to coords
// loopIndex: current loop index
// pointCount: number of points to use to estimate RT
// indices: indices of the pointCount points in d_coord
// d_Rt_relative: resulting relative transformation
__host__ __device__ void estimateRigidTransform3D(const float* d_coord, int loopIndex, 
  int pointCount, int *indices, float* d_Rt_relative) {

  // We are using 3 three-dimensional points to estimate rigid transformation.
  // NOTE: we unroll most of the dim loops because we always assume it's 3
  int dim = 3;

  float *x = new float[pointCount * dim];
  float *y = new float[pointCount * dim];

  // Recover ref and mov coords (both in world coordinates) with the intention
  // of transforming mov coords into ref coords

  // Loop through all the points (3 because we're doing three-point
  // correspondance) to get coordinates
  for (int i = 0; i < pointCount; i++) {
    // Get point index
    int pointIndex = indices[i];

    // Get the three random points (ref coords)
    x[i * dim + 0] = d_coord[6 * pointIndex + 0];
    x[i * dim + 1] = d_coord[6 * pointIndex + 1];
    x[i * dim + 2] = d_coord[6 * pointIndex + 2];

    // Get the three random points (mov coords)
    y[i * dim + 0] = d_coord[6 * pointIndex + 3];
    y[i * dim + 1] = d_coord[6 * pointIndex + 4];
    y[i * dim + 2] = d_coord[6 * pointIndex + 5];
  }

  // Compute the centroid of the three random points by taking the average of
  // their coordinates
  float *x_centroid = new float[dim]();
  float *y_centroid = new float[dim]();

  for (int i = 0; i < pointCount; i++) {
    x_centroid[0] += x[i * dim + 0];
    x_centroid[1] += x[i * dim + 1];
    x_centroid[2] += x[i * dim + 2];

    y_centroid[0] += y[i * dim + 0];
    y_centroid[1] += y[i * dim + 1];
    y_centroid[2] += y[i * dim + 2];
  }

  for (int i = 0; i < dim; i++) {
    x_centroid[i] = x_centroid[i] / pointCount;
    y_centroid[i] = y_centroid[i] / pointCount;
  }

  // Get point coordinates relative to centroid
  for (int i = 0; i < pointCount; i++) {
    x[i * dim + 0] -= x_centroid[0];
    x[i * dim + 1] -= x_centroid[1];
    x[i * dim + 2] -= x_centroid[2];

    y[i * dim + 0] -= y_centroid[0];
    y[i * dim + 1] -= y_centroid[1];
    y[i * dim + 2] -= y_centroid[2];
  }

  // R12 = y_centrized - x_centrized;
  // R21 = x_centrized - y_centrized;
  // R22_1 = y_centrized  + x_centrized;
  float *R22_1 = new float[pointCount * dim];
  for (int i = 0; i < pointCount; i++) {
    R22_1[i * dim + 0] = y[i * dim + 0] + x[i * dim + 0];
    R22_1[i * dim + 1] = y[i * dim + 1] + x[i * dim + 1];
    R22_1[i * dim + 2] = y[i * dim + 2] + x[i * dim + 2];
  }

  // R22 = crossTimesMatrix(R22_1(1:3,:));
  float *R22 = new float[3 * pointCount * dim];
  crossTimesMatrix(R22_1, pointCount, R22);

  float B[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      B[i][j] = 0;
    }
  }

  float A[4][4];

  for (int i = 0; i < pointCount; i++) {
    int pointIndex = i * 9;

    A[0][0] = 0;
    A[0][1] = y[i * dim + 0] - x[i * dim + 0];
    A[0][2] = y[i * dim + 1] - x[i * dim + 1];
    A[0][3] = y[i * dim + 2] - x[i * dim + 2];

    A[1][0] = -y[i * dim + 0] + x[i * dim + 0];
    A[1][1] = R22[pointIndex + 0];
    A[1][2] = R22[pointIndex + 1];
    A[1][3] = R22[pointIndex + 2];
    
    A[2][0] = -y[i * dim + 1] + x[i * dim + 1];
    A[2][1] = R22[pointIndex + 3];
    A[2][2] = R22[pointIndex + 4];
    A[2][3] = R22[pointIndex + 5];
    
    A[3][0] = -y[i * dim + 2] + x[i * dim + 2];
    A[3][1] = R22[pointIndex + 6];
    A[3][2] = R22[pointIndex + 7];
    A[3][3] = R22[pointIndex + 8];

    float A_p[4][4];
    transpose4by4(A, A_p);
    float AA_p[4][4];

    multi4by4(A, A_p, AA_p);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        B[j][k] += AA_p[j][k];
      } 
    }
  }

  delete [] R22_1;
  delete [] R22;
  delete [] x;
  delete [] y;

  float S[4] = { 0, 0, 0, 0 };
  float V[4][4] = { 0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0 };

  // SVD computes something slightly different than MATLAB.
  // Col 1: Same
  // Col 2: Negative
  // Col 3: Negative Col 4 from Matlab
  // Col 4: Negative Col 3 from Matlab
  dsvd(B, 4, 4, S, V);

  int ind = 0;
  float minsingularvalue = S[0];
  for (int i = 1; i < 4; i++) {
    if (S[i] < minsingularvalue) {
      minsingularvalue = S[i];
      ind = i;
    }
  }

  float quat[4];
  for (int i = 0; i < 4; i++) {
    quat[i] = V[i][ind];
  }

  float rot[9];
  quat2rot(quat, rot);
  
  float T1[4][4] = { 1, 0, 0, -y_centroid[0],
                     0, 1, 0, -y_centroid[1],
                     0, 0, 1, -y_centroid[2],
                     0, 0, 0, 1 };

  float T2[4][4] = { rot[0], rot[1], rot[2], 0,
                     rot[3], rot[4], rot[5], 0,
                     rot[6], rot[7], rot[8], 0,
                     0, 0, 0, 1 };

  float T3[4][4] = { 1, 0, 0, x_centroid[0],
                     0, 1, 0, x_centroid[1],
                     0, 0, 1, x_centroid[2],
                     0, 0, 0, 1 };

  delete [] x_centroid;
  delete [] y_centroid;

  float T21[4][4];
  multi4by4(T2, T1, T21);

  float T[4][4];
  multi4by4(T3, T21, T);
  
  #if defined(__CUDA_ARCH__)
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        d_Rt_relative[loopIndex * 12 + i * 4 + j] = T[i][j];
      }
    }
  #else
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        d_Rt_relative[i * 4 + j] = T[i][j];
      }
    }
  #endif
    
  return;
}

// cuSIFT extras/rigidTransform.cu
// For 2 point A and B only 
// Aworld = R * Acamera + t
// Bworld = R * Bcamera + t
// y is totally ignore. value doesn't matter
// 
// d_coord: numPtsx6 matrix. First three elements per row stores the indices
// for the random points and the fourth element holds the error (i.e., output)
// d_randPts: 3xnumLoops matrix. First three elements hold indices to coords
// loopIndex: current loop index
// d_Rt: resulting relative transformation
__host__ __device__ void estimateRigidTransform2D(const float* d_coord, const int* d_randPts, 
                                         int loopIndex, float* d_Rt) {
  // read Point A
  int pointIndexA6 = 6 * d_randPts[loopIndex * 3];
  float Pworld_Ax  = d_coord[pointIndexA6];
  float Pworld_Az  = d_coord[pointIndexA6 + 2];
  float Pcamera_Ax = d_coord[pointIndexA6 + 3];
  float Pcamera_Az = d_coord[pointIndexA6 + 5];

  // read Point B
  int pointIndexB6 = 6 * d_randPts[loopIndex * 3 + 1];
  float Pworld_Bx  = d_coord[pointIndexB6];
  float Pworld_Bz  = d_coord[pointIndexB6 + 2];
  float Pcamera_Bx = d_coord[pointIndexB6 + 3];
  float Pcamera_Bz = d_coord[pointIndexB6 + 5];

  // get the difference vector and normalize it to a unit vector

  // world
  float DXworld = Pworld_Ax - Pworld_Bx;  // the difference vector
  float DZworld = Pworld_Az - Pworld_Bz;  // the difference vector
  float LXZworld = sqrt(DXworld * DXworld + DZworld * DZworld); // the norm
  float DXworld_norm = DXworld / LXZworld;  // unit vector
  float DZworld_norm = DZworld / LXZworld;  // unit vector

  // camera
  float DXcamera = Pcamera_Ax - Pcamera_Bx; // the difference vector
  float DZcamera = Pcamera_Az - Pcamera_Bz; // the difference vector
  float LXZcamera = sqrt(DXcamera * DXcamera + DZcamera * DZcamera);  // the norm
  float DXcamera_norm = DXcamera / LXZcamera; // unit vector
  float DZcamera_norm = DZcamera / LXZcamera; // unit vector

  // rotation angle
  float cosAngle = DXworld_norm * DXcamera_norm + DZworld_norm * DZcamera_norm;
  float sinAngle = DZworld_norm * DXcamera_norm - DXworld_norm * DZcamera_norm; 

  // sum of the two points

  // world
  float SXworld = Pworld_Ax + Pworld_Bx;
  float SZworld = Pworld_Az + Pworld_Bz;

  // camera
  float SXcamera = Pcamera_Ax + Pcamera_Bx;
  float SZcamera = Pcamera_Az + Pcamera_Bz;

  // writing results
  int baseIdx = loopIndex * 12;

  // rotation
  d_Rt[baseIdx + 0] = cosAngle;
  d_Rt[baseIdx + 1] = 0;
  d_Rt[baseIdx + 2] = -sinAngle;

  d_Rt[baseIdx + 4] = 0;
  d_Rt[baseIdx + 5] = 1;
  d_Rt[baseIdx + 6] = 0;

  d_Rt[baseIdx + 8] = sinAngle;
  d_Rt[baseIdx + 9] = 0;
  d_Rt[baseIdx + 10] = cosAngle;

  // translation
  d_Rt[baseIdx + 3] = (SXworld - cosAngle * SXcamera + sinAngle * SZcamera) / 2;
  d_Rt[baseIdx + 7] = 0;
  d_Rt[baseIdx + 11] = (SZworld - sinAngle * SXcamera - cosAngle * SZcamera) / 2;

  return;
}

__device__ void testRigidTransform(float *d_coord, float *d_Rt_relative, int loopIndex,
                                   int *d_counts, int numPts, float thresh2, char *d_inliers) {
  // Get Rt_relative for this thread
  float Rt_relative[12];
  for (int i = 0; i < 12; i++) {
    Rt_relative[i] = d_Rt_relative[loopIndex * 12 + i];
  }

  // Initialize number of inliers
  int numInliers = 0;

  // Loop through all the matched points
  for (int i = 0; i < numPts; i++) {
    float x1 = d_coord[6 * i + 0];
    float y1 = d_coord[6 * i + 1];
    float z1 = d_coord[6 * i + 2];
    float x2 = d_coord[6 * i + 3];
    float y2 = d_coord[6 * i + 4];
    float z2 = d_coord[6 * i + 5];

    // Apply Rt to each point
    float xt = Rt_relative[0] * x2 + Rt_relative[1] * y2 + Rt_relative[2] * z2 + Rt_relative[3];
    float yt = Rt_relative[4] * x2 + Rt_relative[5] * y2 + Rt_relative[6] * z2 + Rt_relative[7];
    float zt = Rt_relative[8] * x2 + Rt_relative[9] * y2 + Rt_relative[10] * z2 + Rt_relative[11];
    float err = (xt - x1) * (xt - x1) + (yt - y1) * (yt - y1) + (zt - z1) * (zt - z1);

    // If the error is less than the threshhold, increment the inlier count
    if (err < thresh2) {
      numInliers++;
      d_inliers[loopIndex * numPts + i] = 1;
    } 
  }

  // Technically we shouldn't overwrite; we're also writing numLoops-major
  // order whereas the original randPts array was numPts-major
  d_counts[loopIndex] = numInliers;
  return;
}

__global__ void EstimateRigidTransformD(float *d_coord, float *d_Rt_relative, int *d_indices,
                                        int numLoops, int numPts, float thresh2, 
                                        RigidTransformType type, curandState_t *d_states, char *d_inliers) {
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int loopIndex = blockDim.x * bx + tx;

  if (loopIndex >= numLoops) return;

  // If d_states is not NULL, then 
  if (d_states != NULL) {
    curandState_t *state = d_states + loopIndex;
    int p1 = curand(state) % numPts;
    int p2 = curand(state) % numPts;
    int p3 = curand(state) % numPts;

    // Make sure they are all unique
    while (p2 == p1) p2 = curand(state) % numPts;
    while (p3 == p1 || p3 == p2) p3 = curand(state) % numPts;

    // Store the indices
    int d_index = loopIndex * 3;
    d_indices[d_index + 0] = p1;
    d_indices[d_index + 1] = p2;
    d_indices[d_index + 2] = p3;
  }

  // First, estimate the rigid transforms over all the loops
  switch(type) {
    case RigidTransformType2D:
    estimateRigidTransform2D(d_coord, d_indices, loopIndex, d_Rt_relative);
    break;

    case RigidTransformType3D:
    int *indices = d_indices + loopIndex * 3;

    estimateRigidTransform3D(d_coord, loopIndex, 3, indices, d_Rt_relative);

    break;
  }

  // Next, test all the rigid transforms and get inliers
  testRigidTransform(d_coord, d_Rt_relative, loopIndex, d_indices, numPts, thresh2, d_inliers);
}

/* this GPU kernel function is used to initialize the random states */
__global__ void initCURAND(unsigned int seed, curandState_t* d_states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &d_states[blockIdx.x]);
}

// Host function that doesn't use OpenCV Mat
void EstimateRigidTransformH(const float *h_coord, float *Rt_relative, int *numInliers,
                             int numLoops, int numPts, float thresh2, RigidTransformType type, int *h_indices, 
                             char *h_inliers) {

  // Start timer
  TimerGPU timer(0);

  // Allocate device memory

  // Pointer to input coordinates (6xnumPts array)
  float *d_coord;
  safeCall(cudaMalloc((void **)&d_coord, 6 * sizeof(float) * numPts));
  
  // Pointer to point indices to use in each loop
  int *d_indices;
  safeCall(cudaMalloc((void **)&d_indices, 3 * sizeof(int) * numLoops));

  curandState_t *d_states;

  bool delete_h_indices = false;
  if (h_indices == NULL) {
    h_indices = new int[3 * numLoops];
    safeCall(cudaMalloc((void **)&d_states, numLoops * sizeof(curandState_t)));
    initCURAND<<<numLoops, 1>>>(time(0), d_states);
    delete_h_indices = true;
  } else {
    d_states = NULL;
    safeCall(cudaMemcpy(d_indices, h_indices, 3 * sizeof(int) * numLoops, cudaMemcpyHostToDevice));
  }
  
  // Boolean array to see which points are considered inliers for each loop.
  // Using char because need to explore bool behavior more
  bool delete_h_inliers = false;
  if (h_inliers == NULL) {
    h_inliers = new char[numPts];
    delete_h_inliers = true;
  }

  char *d_inliers;
  safeCall(cudaMalloc((void **)&d_inliers, numPts * sizeof(char) * numLoops));
  safeCall(cudaMemset(d_inliers, 0, numPts * sizeof(char) * numLoops));

  // Pointer to Rt computed for each loop
  float *d_Rt_relative;
  safeCall(cudaMalloc((void **)&d_Rt_relative, 12 * sizeof(float) * numLoops));

  // Copy memory from host to device
  safeCall(cudaMemcpy(d_coord, h_coord, 6 * sizeof(float) * numPts, cudaMemcpyHostToDevice));

  // Run ransac to find Rt
  int numBlocks = iDivUp(numLoops, 128);
  EstimateRigidTransformD<<<numBlocks, 128>>>(d_coord, d_Rt_relative, d_indices, numLoops, numPts, thresh2, type, d_states, d_inliers);
  checkMsg("EstimateRigidTransformH() execution failed\n");
  safeCall(cudaThreadSynchronize());

  // Copy results back to host
  safeCall(cudaMemcpy(h_indices, d_indices, sizeof(int) * numLoops, cudaMemcpyDeviceToHost));

  // Find result with max inliers
  int maxIndex = -1, maxCount = -1;
  for (int i= 0; i < numLoops; i++) {
    // fprintf(stderr, "Inliers in loop %d: %d\n", i, h_indices[i]);
    if (h_indices[i] >= maxCount) {
       maxCount = h_indices[i];
       maxIndex = i;
    }
  }
  *numInliers = maxCount;

  // Copy back inlier indices
  safeCall(cudaMemcpy(h_inliers, d_inliers + maxIndex * numPts, numPts * sizeof(char), cudaMemcpyDeviceToHost));

  // Do something very naive for now to extract list of inlier indices
  int *inlierIndices = new int[maxCount];
  int inlierCounter = 0;

  for (int i = 0; i < numPts; i++) {
    if (h_inliers[i] == 1) {
      inlierIndices[inlierCounter] = i;
      inlierCounter++;
    }
  }

  switch(type) {
    case RigidTransformType2D:
    // estimateRigidTransform2D(d_coord, d_randPts, loopIndex, d_Rt_relative);
    break;

    case RigidTransformType3D:
    estimateRigidTransform3D(h_coord, maxIndex, *numInliers, inlierIndices, Rt_relative);
    break;
  }

  if (delete_h_inliers) {
    delete [] h_inliers;
  }

  if (delete_h_indices) {
    delete [] h_indices;
  }
  
  // Clean up
  safeCall(cudaFree(d_Rt_relative));
  safeCall(cudaFree(d_coord));
  safeCall(cudaFree(d_inliers));

  if (h_indices == NULL) {
    safeCall(cudaFree(d_states));
  } else {
    safeCall(cudaFree(d_indices));
  }

#ifdef VERBOSE
  double gpuTime = timer.read();
  printf("FindRigidTransform time =         %.2f ms\n", gpuTime);
  std::cout << std::endl;
  std::cout << "RANSAC Fit Rt" << std::endl;

  for (int i = 0; i < 12; i++) {
    fprintf(stderr, "%0.4f ", Rt_relative[i]);
    if ((i + 1) % 4 == 0) std::cout << std::endl;
  }

  std::cout << std::endl;
  printf("Num loops: %d\n", numLoops);
  printf("Threshold^2 %0.4f\n", thresh2);

  printf("numofMatch = %d \n",numInliers[0]);
#endif

  return;
}

/* Convenience function to use OpenCV mat
 *
 * refCoord: 3xnumPts 3D points in reference frame coordinates
 * movCoord: 3xnumPts 3D points in next frame coordinates
 * Rt_relative: 12 floats relative transform matrix
 * numInliers: number of inliers
 * numLoops: number of iterations to run RANSAC
 * thresh: distance threshhold
*/
void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
                            float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh, RigidTransformType type) {

  // Combine refCoord and movCoord into contiguous block of memory
  cv::Mat coord(refCoord.size().height, refCoord.size().width + movCoord.size().width, CV_32FC1);
  cv::Mat left(coord, cv::Rect(0, 0, refCoord.size().width, refCoord.size().height));
  refCoord.copyTo(left);
  cv::Mat right(coord, cv::Rect(refCoord.size().width, 0, movCoord.size().width, movCoord.size().height));
  movCoord.copyTo(right);
  float *h_coord = (float*)coord.data;
  
  // Number of matches
  int numPts = refCoord.size().height;

  EstimateRigidTransformH(h_coord, Rt_relative, numInliers, numLoops, numPts, thresh * thresh, type);

  return;
}
 
// Convenience function to use vector of SiftMatch objects. Arguments are the same as above.
void EstimateRigidTransform(vector<SiftMatch *> matches, float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh, RigidTransformType type) {
  // Combine data into a contiguous block of memory
  float *h_coord = new float[6 * matches.size()];
  for (int i = 0; i < matches.size(); i++) {
    memcpy(h_coord + 6 * i, matches[i]->pt1->coords3D, sizeof(float) * 3);
    memcpy(h_coord + 6 * i + 3, matches[i]->pt2->coords3D, sizeof(float) * 3);
  }

  EstimateRigidTransformH(h_coord, Rt_relative, numInliers, numLoops, matches.size(), thresh * thresh, type);

  delete [] h_coord;
}