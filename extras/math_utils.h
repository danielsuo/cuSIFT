#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cstdio>

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

__host__ __device__ double PYTHAG(double a, double b);
__host__ __device__ void dsvd(float a[4][4], int m, int n, float w[4], float v[4][4]);
__host__ __device__ void quat2rot(const float Q[4], float R[9]);
__host__ __device__ void crossTimesMatrix(const float *V, int pointCount, float *V_times);
__host__ __device__ void transpose4by4(float a[4][4], float b[4][4]);
__host__ __device__ void multi4by4 (float a[4][4], float b[4][4], float c[4][4]);

#endif