#include <cmath>
#include <cfloat>
#include <stdio.h>
#include <vector>

#include "cuSIFT.h"

#include "gtest/gtest.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

extern "C" {
  #include <vl/sift.h>
}

TEST(Detector, DetectorCUSIFTTest) {
  cv::Mat im = cv::imread("../test/data/color1.jpg", cv::IMREAD_GRAYSCALE);
  im.convertTo(im, CV_32FC1);

  int w = im.size().width;
  int h = im.size().height;

  // // Perform some initial blurring (if needed)
  // cv::GaussianBlur(limg, limg, cv::Size(3, 3), 0.5);
  // cv::GaussianBlur(rimg, rimg, cv::Size(3, 3), 0.5);
        
  // Initial Cuda images and download images to device
  InitCuda(0);
  cuImage cuIm;
  cuIm.Allocate(w, h, (float*)im.data);
  cuIm.Download();

  // Extract Sift features from images
  SiftData siftData;
  float initBlur = 0.0f;
  float thresh = 0.1f;
  int numSift = 4096;
  InitSiftData(siftData, numSift, true, true); 
    
  ExtractSift(siftData, cuIm, 6, initBlur, thresh, 0.0f);

  FILE *fp = fopen("../test/data/cusift1", "wb");
  fwrite(&siftData.numPts, sizeof(uint32_t), 1, fp);

  for (int i = 0; i < siftData.numPts; i++) {
    SiftPoint pt = siftData.h_data[i];
    fwrite(pt.coords2D, sizeof(float), 2, fp);
    fwrite(&pt.scale, sizeof(float), 1, fp);
    fwrite(&pt.orientation, sizeof(float), 1, fp);
  }

  fclose(fp);
}

TEST(Detector, DetectorVLFeatTest) {
  cv::Mat im = cv::imread("../test/data/color1.jpg", cv::IMREAD_GRAYSCALE);
  cv::imwrite("../test/data/gray1.jpg", im);
  im.convertTo(im, CV_32FC1);

  FILE *imdat = fopen("../test/data/gray1", "wb");
  fwrite((float *)im.data, sizeof(float), im.size().height * im.size().width, imdat);
  fclose(imdat);

  // Overview here: http://www.vlfeat.org/api/sift.html
  // Documentation here: http://www.vlfeat.org/api/sift_8c.html
  int width = im.size().width;
  int height = im.size().height;

  int numOctaves = (int)floor(log2(std::min(width, height)));
  int levelsPerOctave = 3;
  int firstOctaveIndex = 0;

  VlSiftFilt *siftFilt = vl_sift_new(width, height, numOctaves,
    levelsPerOctave, firstOctaveIndex);

  // Additional parameters: http://www.vlfeat.org/api/sift_8h.html
  // Influence filtering:
  vl_sift_set_peak_thresh(siftFilt, 0);
  vl_sift_set_edge_thresh(siftFilt, 10);
  vl_sift_set_norm_thresh(siftFilt, 0);
  // Influence descriptor calculation:
  vl_sift_set_magnif(siftFilt, 3);
  vl_sift_set_window_size(siftFilt, 2);

  int result = vl_sift_process_first_octave(siftFilt, (float *)im.data);

  FILE *fp = fopen("../test/data/sift1", "wb");

  while (result != VL_ERR_EOF) {
    fprintf(stderr, "Processing octave %d of %d\n", vl_sift_get_octave_index(siftFilt), vl_sift_get_noctaves(siftFilt));
    vl_sift_detect(siftFilt);
    VlSiftKeypoint const *keypoints = vl_sift_get_keypoints(siftFilt);
    int numKeypoints = vl_sift_get_nkeypoints(siftFilt);

    for (int i = 0; i < numKeypoints; i++) {
      double angles[4];
      int numOrientations = vl_sift_calc_keypoint_orientations(siftFilt, angles, keypoints + i);

      for (int j = 0; j < numOrientations; j++) {
        float desc[128];
        vl_sift_calc_keypoint_descriptor(siftFilt, desc, keypoints + i, angles[j]);

        float angle = (float)angles[j];
        fwrite(&keypoints[i].x, sizeof(float), 1, fp);
        fwrite(&keypoints[i].y, sizeof(float), 1, fp);
        fwrite(&keypoints[i].sigma, sizeof(float), 1, fp);
        fwrite(&angle, sizeof(float), 1, fp);
      }
    }

    result = vl_sift_process_next_octave(siftFilt);
  }

  fclose(fp);

  vl_sift_delete(siftFilt);
  im.release();
}