#include "cudautils.h"
#include "rigidTransform.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define max(x,y) ((x)>(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

__device__ double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}

__device__ void dsvd(float a[4][4], int m, int n, float w[4], float v[4][4])
{
  /*
   * Input to dsvd is as follows:
   *   a = mxn matrix to be decomposed, gets overwritten with u
   *   m = row dimension of a
   *   n = column dimension of a
   *   w = returns the vector of singular values of a
   *   v = returns the right orthogonal transformation matrix
   *   http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
  */
  int flag, i, its, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  double rv1[4];

  if (m < n) {
    printf("#rows must be > #cols \n");
    return;
  }

  /* Householder reduction to bidiagonal form */
  for (i = 0; i < n; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) 
        scale += fabs((double)a[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          a[k][i] = (float)((double)a[k][i]/scale);
          s += ((double)a[k][i] * (double)a[k][i]);
        }
        f = (double)a[i][i];
                //SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][i] = (float)(f - g);
        if (i != n - 1) {
          for (j = l; j < n; j++) {
            for (s = 0.0, k = i; k < m; k++) 
              s += ((double)a[k][i] * (double)a[k][j]);
            f = s / h;
            for (k = i; k < m; k++) 
              a[k][j] += (float)(f * (double)a[k][i]);
          }
        }
        for (k = i; k < m; k++) 
          a[k][i] = (float)((double)a[k][i]*scale);
      }
    }
    w[i] = (float)(scale * g);

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != n - 1) {
      for (k = l; k < n; k++) 
        scale += fabs((double)a[i][k]);
      if (scale) {
        for (k = l; k < n; k++) {
          a[i][k] = (float)((double)a[i][k]/scale);
          s += ((double)a[i][k] * (double)a[i][k]);
        }
        f = (double)a[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][l] = (float)(f - g);
        for (k = l; k < n; k++) 
          rv1[k] = (double)a[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < n; k++) 
              s += ((double)a[j][k] * (double)a[i][k]);
            for (k = l; k < n; k++) 
              a[j][k] += (float)(s * rv1[k]);
          }
        }
        for (k = l; k < n; k++) 
          a[i][k] = (float)((double)a[i][k]*scale);
      }
    }
    anorm = max(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
  }
  
    /* accumulate the right-hand transformation */
  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
        for (j = l; j < n; j++)
          v[j][i] = (float)(((double)a[i][j] / (double)a[i][l]) / g);
                    /* double division to avoid underflow */
        for (j = l; j < n; j++) {
          for (s = 0.0, k = l; k < n; k++) 
            s += ((double)a[i][k] * (double)v[k][j]);
          for (k = l; k < n; k++) 
            v[k][j] += (float)(s * (double)v[k][i]);
        }
      }
      for (j = l; j < n; j++) 
        v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }
  
    /* accumulate the left-hand transformation */
  for (i = n - 1; i >= 0; i--) {
    l = i + 1;
    g = (double)w[i];
    if (i < n - 1) 
      for (j = l; j < n; j++) 
        a[i][j] = 0.0;
      if (g) {
        g = 1.0 / g;
        if (i != n - 1) {
          for (j = l; j < n; j++) {
            for (s = 0.0, k = l; k < m; k++) 
              s += ((double)a[k][i] * (double)a[k][j]);
            f = (s / (double)a[i][i]) * g;
            for (k = i; k < m; k++) 
              a[k][j] += (float)(f * (double)a[k][i]);
          }
        }
        for (j = i; j < m; j++) 
          a[j][i] = (float)((double)a[j][i]*g);
      }
      else {
        for (j = i; j < m; j++) 
          a[j][i] = 0.0;
      }
      ++a[i][i];
    }

    /* diagonalize the bidiagonal form */
    /* loop over singular values */
    for (k = n - 1; k >= 0; k--) {  
    /* loop over allowed iterations */    
      for (its = 0; its < 30; its++) {
        flag = 1;

        for (l = k; l >= 0; l--) {       
          nm = l - 1;
          if (fabs(rv1[l]) + anorm == anorm) {
            flag = 0;
            break;
          }
          if (fabs((double)w[nm]) + anorm == anorm) 
            break;
        }
        if (flag) 
        {
          c = 0.0;
          s = 1.0;
          for (i = l; i <= k; i++) 
          {
            f = s * rv1[i];
            if (fabs(f) + anorm != anorm) 
            {
              g = (double)w[i];
              h = PYTHAG(f, g);
              w[i] = (float)h; 
              h = 1.0 / h;
              c = g * h;
              s = (- f * h);
              for (j = 0; j < m; j++) 
              {
                y = (double)a[j][nm];
                z = (double)a[j][i];
                a[j][nm] = (float)(y * c + z * s);
                a[j][i] = (float)(z * c - y * s);
              }
            }
          }
        }
        z = (double)w[k];
        if (l == k) 
            {                  /* convergence */
          if (z < 0.0) 
                {              /* make singular value nonnegative */
            w[k] = (float)(-z);
          for (j = 0; j < n; j++) 
            v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
                //free((void*) rv1);
        printf("No convergence after 30,000! iterations \n");
        return;
      }

            /* shift from bottom 2 x 2 minor */
      x = (double)w[l];
      nm = k - 1;
      y = (double)w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

            /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) 
      {
        i = j + 1;
        g = rv1[i];
        y = (double)w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < n; jj++) 
        {
          x = (double)v[jj][j];
          z = (double)v[jj][i];
          v[jj][j] = (float)(x * c + z * s);
          v[jj][i] = (float)(z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = (float)z;
        if (z) 
        {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) 
        {
          y = (double)a[jj][j];
          z = (double)a[jj][i];
          a[jj][j] = (float)(y * c + z * s);
          a[jj][i] = (float)(z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = (float)x;
    }
  }
    //free((void*) rv1);
  return;
}

__device__ void quat2rot(const float Q[4], float R[9])
{
  /*  QUAT2ROT */
  /*    R = QUAT2ROT(Q) converts a quaternion (4x1 or 1x4) into a rotation mattrix */
  R[0] = ((Q[0] * Q[0] + Q[1] * Q[1]) - Q[2] * Q[2]) - Q[3] * Q[3];
  R[1] = 2.0 * (Q[1] * Q[2] - Q[0] * Q[3]);
  R[2] = 2.0 * (Q[1] * Q[3] + Q[0] * Q[2]);
  R[3] = 2.0 * (Q[1] * Q[2] + Q[0] * Q[3]);
  R[4] = ((Q[0] * Q[0] - Q[1] * Q[1]) + Q[2] * Q[2]) - Q[3] * Q[3];
  R[5] = 2.0 * (Q[2] * Q[3] - Q[0] * Q[1]);
  R[6] = 2.0 * (Q[1] * Q[3] - Q[0] * Q[2]);
  R[7] = 2.0 * (Q[2] * Q[3] + Q[0] * Q[1]);
  R[8] = ((Q[0] * Q[0] - Q[1] * Q[1]) - Q[2] * Q[2]) + Q[3] * Q[3];

  return;
}

__device__ void crossTimesMatrix(const float V[9], int V_length,float V_times[3][3][3])
{
  //V a 3xN matrix, rpresenting a series of 3x1 vectors
  for (int i = 0; i < V_length; i++) {
    V_times[0][0][i] = 0;
    V_times[0][1][i] = -V[2 + 3 * i];
    V_times[0][2][i] =  V[1 + 3 * i];

    V_times[1][0][i] =  V[2 + 3 * i];
    V_times[1][1][i] = 0;
    V_times[1][2][i] = -V[0 + 3 * i];

    V_times[2][0][i] = -V[1 + 3 * i];
    V_times[2][1][i] =  V[0 + 3 * i];
    V_times[2][2][i] = 0;
  }

  return;
}

__device__ void transpose4by4(float a[4][4], float b[4][4]) {
  for (int i = 0;i<4;i++) {
    for (int j= 0; j<4; j++) {
      b[j][i] = a[i][j];
    }
  }
  return;
}

__device__ void multi4by4 (float a[4][4],float b[4][4],float c[4][4]) {
  for (int i = 0; i <4 ; i++) {
    for (int j= 0 ; j<4 ; j++) {
      c[i][j]= 0;
      for (int k= 0 ; k<4; k++) {
        c[i][j]=c[i][j]+a[i][k]*b[k][j];
      }
    }
  }
  return;
}

// For 3 point only 
// X: [x0, y0, z0, x1, y1, z1, x2, y2, z2] 
// Y: [x0',y0',z0',x1,y1',z1',x2',y2',z2'] 
// xh = T * yh
// 
// d_coord: numPtsx6 matrix. First three elements per row stores the indices
// for the random points and the fourth element holds the error (i.e., output)
// d_randPts: 4xnumLoops matrix. First three elements hold indices to coords
// idx: current index
// numLoops: number of RANSAC loops
// d_Rt_relative: resulting relative transformation
__device__ void estimateRigidTransform(const float* d_coord, const int* d_randPts, 
                                       int idx, int numLoops, float* d_Rt_relative) {

  // form the 3x3 point matrix
  int pointCount = 3;

  // Recover ref and mov coords (both in world coordinates) with the intention
  // of transforming mov coords into ref coords
  float x_in[9];
  float y_in[9];

  // Loop through all the points (3 because we're doing three-point
  // correspondance) to get coordinates
  for (int i = 0; i < pointCount; i++) {
    // Get point index
    int pointIndex = d_randPts[i * numLoops + idx];

    // Get the three random points (ref coords)
    x_in[i * 3 + 0] = d_coord[6 * pointIndex + 0];
    x_in[i * 3 + 1] = d_coord[6 * pointIndex + 1];
    x_in[i * 3 + 2] = d_coord[6 * pointIndex + 2];

    // Get the three random points (mov coords)
    y_in[i * 3 + 0] = d_coord[6 * pointIndex + 3];
    y_in[i * 3 + 1] = d_coord[6 * pointIndex + 4];
    y_in[i * 3 + 2] = d_coord[6 * pointIndex + 5];
  }

  // Compute the centroid of the three random points by taking the average of
  // their coordinates
  float x_centroid[3] = {0.0, 0.0, 0.0};
  float y_centroid[3] = {0.0, 0.0, 0.0};

  for (int i = 0;i<pointCount;i++) {
    x_centroid[0] += x_in[i * 3 + 0];
    x_centroid[1] += x_in[i * 3 + 1];
    x_centroid[2] += x_in[i * 3 + 2];

    y_centroid[0] += y_in[i * 3 + 0];
    y_centroid[1] += y_in[i * 3 + 1];
    y_centroid[2] += y_in[i * 3 + 2];
  }

  for (int i = 0;i<3;i++) {
    x_centroid[i] = x_centroid[i]/pointCount;
    y_centroid[i] = y_centroid[i]/pointCount;
  }

  // Get point coordinates relative to centroid
  float x[9], y[9];
  for (int i = 0; i < pointCount; i++) {
    x[0 + i * 3] = x_in[0 + i * 3] - x_centroid[0];
    x[1 + i * 3] = x_in[1 + i * 3] - x_centroid[1];
    x[2 + i * 3] = x_in[2 + i * 3] - x_centroid[2];

    y[0 + i * 3] = y_in[0 + i * 3] - y_centroid[0];
    y[1 + i * 3] = y_in[1 + i * 3] - y_centroid[1];
    y[2 + i * 3] = y_in[2 + i * 3] - y_centroid[2];
  }

  float R12[9];
  for (int i = 0; i < pointCount; i++) {
      R12[0 + i * 3] = y[0 + i * 3] - x[0 + i * 3];
      R12[1 + i * 3] = y[1 + i * 3] - x[1 + i * 3];
      R12[2 + i * 3] = y[2 + i * 3] - x[2 + i * 3];
  }

  float R21[9];
  for (int i = 0; i < pointCount; i++) {
      R21[0 + i * 3] = - y[0 + i * 3] + x[0 + i * 3];
      R21[1 + i * 3] = - y[1 + i * 3] + x[1 + i * 3];
      R21[2 + i * 3] = - y[2 + i * 3] + x[2 + i * 3];
  }

  float R22_1[9];
  for (int i = 0; i < pointCount; i++) {
      R22_1[0 + i * 3] = y[0 + i * 3] + x[0 + i * 3];
      R22_1[1 + i * 3] = y[1 + i * 3] + x[1 + i * 3];
      R22_1[2 + i * 3] = y[2 + i * 3] + x[2 + i * 3];
  }

  float R22[3][3][3];
  crossTimesMatrix(R22_1, pointCount, R22);
  float B[4][4];
  for (int i = 0;i<4;i++) {
     for (int j = 0;j<4;j++) {
        B[i][j] = 0;
     }
  }

  float A[4][4];

  for (int i = 0; i < pointCount; i++) {
    A[0][0] = 0;
    A[0][1] = R12[0 + i * 3];
    A[0][2] = R12[1 + i * 3];
    A[0][3] = R12[2 + i * 3];

    A[1][0] = R21[0 + i * 3];
    A[1][1] = R22[0][0][i];
    A[1][2] = R22[0][1][i];
    A[1][3] = R22[0][2][i];

    A[2][0] = R21[1 + i * 3];
    A[2][1] = R22[1][0][i];
    A[2][2] = R22[1][1][i];
    A[2][3] = R22[1][2][i];

    A[3][0] = R21[2 + i * 3];
    A[3][1] = R22[2][0][i];
    A[3][2] = R22[2][1][i];
    A[3][3] = R22[2][2][i];

    float A_p[4][4];
    transpose4by4(A,A_p);
    float AA_p[4][4];
    multi4by4(A,A_p,AA_p);

    for (int j = 0;j<4;j++) {
      for (int k = 0;k<4;k++) {
        B[j][k] = B[j][k]+AA_p[j][k];
      } 
    }
  }

  float S[4] = {0, 0, 0, 0};
  float V[4][4] = {0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0};

  dsvd(B,4,4,S,V);

  int ind = 0;
  float minsingularvalue = S[0];
  for (int i= 0;i<4;i++) {
    if (S[i]<minsingularvalue) {
      minsingularvalue = S[i];
      ind =i;
    }
  }

  float quat[4];
  for (int i = 0;i<4;i++) {
    quat[i] = V[i][ind];
  }

  float rot[9];
  quat2rot(quat,rot);
  
  float T1[4][4] = {1,0,0,-y_centroid[0],
                   0,1,0,-y_centroid[1],
                   0,0,1,-y_centroid[2],
                   0,0,0,1};
  float T2[4][4] = {rot[0],rot[1],rot[2],0,
                   rot[3],rot[4],rot[5],0,
                   rot[6],rot[7],rot[8],0,
                   0,0,0,1};
  float T3[4][4] = {1,0,0,x_centroid[0],
                   0,1,0,x_centroid[1],
                   0,0,1,x_centroid[2],
                   0,0,0,1};

  float T21[4][4];
  multi4by4(T2,T1,T21);

  float T[4][4];
  multi4by4(T3,T21,T);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      d_Rt_relative[idx * 12 + i * 4 + j] = T[i][j];
    }
  }

  return;
}

__device__ void testRigidTransform(float *d_coord, float *d_Rt_relative, int idx,
                                   int *d_counts, int numPts, float thresh2) {
  // Get Rt_relative for this thread
  float Rt_relative[12];
  for (int i = 0; i < 12; i++) {
    Rt_relative[i] = d_Rt_relative[idx * 12 + i];
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
    }
  }

  // Technically we shouldn't over
  d_counts[idx] = numInliers;
  return;
}

__global__ void ComputeRigidTransform(float *d_coord, float *d_Rt_relative,
                                      int *d_randPts, int numPts, float thresh2) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int idx = blockDim.x * bx + tx;
  const int numLoops = blockDim.x * gridDim.x;

  // First, estimate the rigid transforms over all the loops
  estimateRigidTransform(d_coord, d_randPts, idx, numLoops, d_Rt_relative);

  // Next, test all the rigid transforms and make a choice
  testRigidTransform(d_coord, d_Rt_relative, idx, d_randPts, numPts, thresh2);
}

void FindRigidTransform(const float *h_coord, int *h_randPts, float *Rt_relative, int *numInliers,
                        int numLoops, int numPts, float thresh2) {
  // Start timer
  TimerGPU timer(0);

  // Pointers to device memory
  float *d_coord, *d_Rt_relative;
  int *d_randPts;

  // Allocate device memory
  safeCall(cudaMalloc((void **)&d_coord, 6*sizeof(float)*numPts));
  safeCall(cudaMalloc((void **)&d_randPts, 3*sizeof(int)*numLoops));
  safeCall(cudaMalloc((void **)&d_Rt_relative, 12*sizeof(float)*numLoops));
  
  // Copy memory from host to device
  safeCall(cudaMemcpy(d_randPts, h_randPts, 3*sizeof(int)*numLoops, cudaMemcpyHostToDevice));
  safeCall(cudaMemcpy(d_coord, h_coord, 6*sizeof(float)*numPts, cudaMemcpyHostToDevice));

  // Run ransac to find Rt
  ComputeRigidTransform<<<numLoops/128, 128>>>(d_coord, d_Rt_relative, d_randPts, numPts, thresh2);
  checkMsg("ComputeHomographies() execution failed\n");
  safeCall(cudaThreadSynchronize());

  // Copy results back to host
  safeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, cudaMemcpyDeviceToHost));
  
  int maxIndex = -1, maxCount = -1;
  for (int i= 0; i < numLoops; i++) {
    if (h_randPts[i] > maxCount) {
       maxCount = h_randPts[i];
       maxIndex = i;
    }
  }
  *numInliers = maxCount;

  // Copy most likely relative Rt
  safeCall(cudaMemcpy(Rt_relative, &d_Rt_relative[12*maxIndex], 12*sizeof(float), cudaMemcpyDeviceToHost));
  
  // Clean up
  safeCall(cudaFree(d_Rt_relative));
  safeCall(cudaFree(d_randPts));
  safeCall(cudaFree(d_coord));
  
  #ifdef VERBOSE
    double gpuTime = timer.read();
    printf("FindRigidTransform time =         %.2f ms\n", gpuTime);
  #endif
}

/* Convenience function to use OpenCV mat
 *
 * refCoord: 3xnumPts 3D points in reference frame coordinates
 * movCoord: 3xnumPts 3D points in next frame coordinates
 * Rt_relative: 12 floats relative transform matrix
 * numInliers: number of inliers
 * numLoops: number of iterations to run RANSAC
 * thresh: distance threshhold
 *
 * TODO:
 * - Naming scheme is confusing (Estimate, Find, Compute Rigid Transform?)
*/
void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
                            float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh) {

  // Combine refCoord and movCoord into contiguous block of memory
  cv::Mat coord(refCoord.size().height, refCoord.size().width + movCoord.size().width, CV_32FC1);
  cv::Mat left(coord, cv::Rect(0, 0, refCoord.size().width, refCoord.size().height));
  refCoord.copyTo(left);
  cv::Mat right(coord, cv::Rect(refCoord.size().width, 0, movCoord.size().width, movCoord.size().height));
  movCoord.copyTo(right);
  float *h_coord = (float*)coord.data;
  
  // Number of matches
  int numPts = refCoord.size().height;
  
  // First three elements per row stores the indices for the random points
  int* h_randPts = (int*)malloc(3 * sizeof(int) * numLoops);
  
  // Choose three random points (their indices) for each iteration
  for (int i = 0; i < numLoops; i++) {
    int p1 = rand() % numPts;
    int p2 = rand() % numPts;
    int p3 = rand() % numPts;

    // Make sure they are all unique
    while (p2 == p1) p2 = rand() % numPts;
    while (p3 == p1 || p3 == p2) p3 = rand() % numPts;

    // Store the indices
    h_randPts[i + 0 * numLoops] = p1;
    h_randPts[i + 1 * numLoops] = p2;
    h_randPts[i + 2 * numLoops] = p3;
  }

  FindRigidTransform(h_coord, h_randPts, Rt_relative, numInliers, numLoops, numPts, thresh * thresh);

  free(h_randPts);
  
#ifdef VERBOSE
  std::cout << std::endl;
  std::cout << "RANSAC Fit Rt" << std::endl;

  for (int i = 0; i < 12; i++) {
    fprintf(stderr, "%0.4f ", Rt_relative[i]);
    if ((i + 1) % 4 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  printf("Num loops: %d\n", numLoops);
  printf("Threshold %0.4f\n", thresh);

  printf("numofMatch = %d \n",numInliers[0]);
#endif

  return;
}
 