#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "cuda_runtime_api.h"
#include "cudaSift.h"
#include "cudautils.h"

using namespace std;

void writeMatToFile(cv::Mat& m, const char* filename);

void PrintSiftData(SiftData &data);
cv::Mat PrintMatchData(SiftData &siftData1, SiftData &siftData2, cv::Mat limg, cv::Mat rimg);
void PrintMatchSiftData(SiftData &siftData1, const char* filename, int imgw);

void ReadVLFeatSiftData(SiftData &siftData, const char *filename);
void ReadMATLABMatchData(cv::Mat &curr_match, cv::Mat &next_match, const char *filename);
void ReadMATLABRt(float *Rt_relative, const char *filename);

void AddSiftData(SiftData &data, SiftPoint *h_data, int numPts);

#endif
