#include "gtest/gtest.h"

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cuImage.h"
#include "cuSIFT.h"

#include "extras/rigidTransform.h"
#include "extras/matching.h"
#include "extras/homography.h"
#include "extras/debug.h"

TEST(Matching, MatchingTest) {
  SiftData siftData1, siftData2;
  ReadVLFeatSiftData(siftData1, "../test/data/sift/sift1");
  ReadVLFeatSiftData(siftData2, "../test/data/sift/sift2");

  int numMatches = ReadMATLABMatchIndices("../test/data/match_indices/match_indices1_2");
  uint32_t *indices_i = new uint32_t[numMatches];
  uint32_t *indices_j = new uint32_t[numMatches];
  numMatches = ReadMATLABMatchIndices("../test/data/match_indices/match_indices1_2", indices_i, indices_j);

  vector<SiftMatch *> matches = MatchSiftData(siftData1, siftData2, MatchSiftDistanceL2);

  for (int i = 0; i < numMatches; i++) {
    // fprintf(stderr, "Match for %d: matlab %d, cusift %d\n", indices_i[i], indices_j[i], siftData1.h_data[indices_i[i] - 1].match + 1);
    EXPECT_EQ(indices_j[i], matches[indices_i[i] - 1]->pt1->match + 1);
  }

  vector<SiftMatch *> mlmatches = ReadMATLABMatchData("../test/data/match/match1_2");

  cerr << "Length of matches: " << matches.size() << endl;
  cerr << "Length of mlmatches: " << mlmatches.size() << endl;
}

TEST(Matching, MatchingRatioTest) {
  SiftData siftData1, siftData2;
  ReadVLFeatSiftData(siftData1, "../test/data/sift/sift1");
  ReadVLFeatSiftData(siftData2, "../test/data/sift/sift2");
  vector<SiftMatch *> matches = MatchSiftData(siftData1, siftData2, MatchSiftDistanceL2, 1000, 0.6);

  cerr << "Originally had " << min(siftData1.numPts, siftData2.numPts) << " matches, now have " << matches.size() << endl;
  EXPECT_EQ(340, matches.size());
}

TEST(Matching, MatchingImageBoundaryTest) {

}

TEST(RigidTransform, RANSACWithIndices) {
  vector<int> indices;
  float Rt[12];
  vector<SiftMatch *> matches = ReadMATLABRANSAC("../test/data/RigidTransform_RANSAC.bin", indices, Rt);

  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 4; j++) {
  //     fprintf(stderr, "%0.4f ", Rt[i * 4 + j]);
  //   }
  //   fprintf(stderr, "\n");
  // }

  int numLoops = indices.size() / 3;

  for (int i = 0; i < numLoops; i++) {
    for (int j = 0; j < 3; j++) {
      fprintf(stderr, "% 4d ", indices[i * 3 + j]);
    }
    fprintf(stderr, "\n");
  }

  float Rt_test[12];
  int numInliers[0];
  float thresh2 = 0.05 * 0.05;

  float *h_coord = new float[6 * matches.size()];
  
  for (int i = 0; i < matches.size(); i++) {
    memcpy(h_coord + 6 * i, matches[i]->pt1->coords3D, sizeof(float) * 3);
    memcpy(h_coord + 6 * i + 3, matches[i]->pt2->coords3D, sizeof(float) * 3);
  }

  char *h_inliers = new char[matches.size()];

  EstimateRigidTransformH(h_coord, Rt_test, numInliers, numLoops, matches.size(), thresh2, RigidTransformType3D, &indices[0], h_inliers);
  // for (int i = 0; i < matches.size(); i++) {
  //   fprintf(stderr, "%d: %d\n", i + 1, h_inliers[i]);
  // }

  fprintf(stderr, "Inliers / total: %d / %lu = %0.4f%%\n", numInliers[0], matches.size(), (float)numInliers[0] / matches.size() * 100);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      fprintf(stderr, "%0.4f ", Rt_test[i * 4 + j]);
    }
    fprintf(stderr, "\n");
  }

  for (int i = 0; i < matches.size(); i++) {
    delete matches[i]->pt1;
    delete matches[i]->pt2;
    delete matches[i];
  }
}

TEST(RigidTransform, RANSACWithRandom) {
  vector<int> indices;
  float Rt[12];
  vector<SiftMatch *> matches = ReadMATLABRANSAC("../test/data/RigidTransform_RANSAC.bin", indices, Rt);
  int numLoops = 4096;
  float Rt_test[12];
  int numInliers[0];

  EstimateRigidTransform(matches, Rt_test, numInliers, numLoops, 0.05, RigidTransformType3D);
  fprintf(stderr, "Inliers / total: %d / %lu = %0.4f%%\n", numInliers[0], matches.size(), (float)numInliers[0] / matches.size() * 100);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      fprintf(stderr, "%0.4f ", Rt_test[i * 4 + j]);
    }
    fprintf(stderr, "\n");
  }
}

TEST(RigidTransform, RANSACTestImage) {
  string limgPath = "../test/data/color1.jpg";
  string rimgPath = "../test/data/color2.jpg";

  // Read images using OpenCV
  cv::Mat limg, rimg;
  cv::imread(limgPath, 0).convertTo(limg, CV_32FC1);
  cv::imread(rimgPath, 0).convertTo(rimg, CV_32FC1);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  cout << "Image size = (" << w << "," << h << ")" << endl;

  // Perform some initial blurring (if needed)
  cv::GaussianBlur(limg, limg, cv::Size(3, 3), 0.5);
  cv::GaussianBlur(rimg, rimg, cv::Size(3, 3), 0.5);
        
  // Initial Cuda images and download images to device
  cout << "Initializing data..." << endl;
  InitCuda(0);
  cuImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 0.0f;
  float thresh = 0.1f;
  int numSift = 4096;
  InitSiftData(siftData1, numSift, true, true); 
  InitSiftData(siftData2, numSift, true, true);
  
  ExtractSift(siftData1, img1, 6, initBlur, thresh, 0.0f);
  ExtractSift(siftData2, img2, 6, initBlur, thresh, 0.0f);

  vector<SiftMatch *> matches = MatchSiftData(siftData1, siftData2, MatchSiftDistanceL2);
  int numLoops = 1024;
  float Rt_test[12];
  int numMatches[0];

  // EstimateRigidTransform(matches, Rt_test, numMatches, numLoops, 0.05, RigidTransformType3D);
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 4; j++) {
  //     fprintf(stderr, "%0.4f ", Rt_test[i * 4 + j]);
  //   }
  //   fprintf(stderr, "\n");
  // }
}

// // The fixture for testing class Project1. From google test primer.
// class Project1Test : public ::testing::Test {
// protected:
//   // You can remove any or all of the following functions if its body
//   // is empty.

//   Project1Test() {
//     // You can do set-up work for each test here.
//   }

//   virtual ~Project1Test() {
//     // You can do clean-up work that doesn't throw exceptions here.
//   }

//   // If the constructor and destructor are not enough for setting up
//   // and cleaning up each test, you can define the following methods:
//   virtual void SetUp() {
//     // Code here will be called immediately after the constructor (right
//     // before each test).
//   }

//   virtual void TearDown() {
//     // Code here will be called immediately after each test (right
//     // before the destructor).
//   }

//   // Objects declared here can be used by all tests in the test case for Project1.
//   Project1 p;
// };

// // Test case must be called the class above
// // Also note: use TEST_F instead of TEST to access the test fixture (from google test primer)
// TEST_F(Project1Test, MethodBarDoesAbc) {
//   int i = 0;
//   p.foo(i); // we have access to p, declared in the fixture
//   EXPECT_EQ(1, i);