#ifndef RGBD_H
#define RGBD_H

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "cudaSift.h"
#ifdef CPURANSAC
#include "estimateRTCPU.h"
#endif
using namespace std;

////////////////////////////////////////////////////////////////////////////////

typedef struct Camera {
  float cx,cy,fx,fy;
  float Kdepth[3][3];
  float R_d2c[3][3];
  float T_d2c[3];
} Camera;



////////////////////////////////////////////////////////////////////////////////

void writeMatToFile(cv::Mat& m, const char* filename);
void EstimateRigidTransform(const cv::Mat refCoord, const cv::Mat movCoord, 
                            float* Rt_relative, int* numInliers, 
                            int numLoops, float thresh);

unsigned int uchar2uint(unsigned char* in);
void uint2uchar(unsigned int in, unsigned char* out);

cv::Mat PrintMatchData(SiftData &siftData1, SiftData &siftData2, cv::Mat limg, cv::Mat rimg);
void PrintMatchSiftData(SiftData &siftData1, const char* filename, int imgw);

void ReadVLFeatSiftData(SiftData &siftData, const char *filename);
void ReadMATLABMatchData(cv::Mat &curr_match, cv::Mat &next_match, const char *filename);
void ReadMATLABRt(float *Rt_relative, const char *filename);

#endif
