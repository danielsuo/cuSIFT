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

#include "extras/matching.h"
#include "extras/homography.h"
#include "extras/debug.h"

using namespace std;

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, cuImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

void demo(int argc, char **argv);
void testMatchingDotProduct(int argc, char **argv);
void testMatchingL2(int argc, char **argv);
void compareMatchingWithMATLAB(int argc, char **argv);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{ 
  // demo(argc, argv);
  // testMatchingDotProduct(argc, argv);
  // testMatchingL2(argc, argv);
  compareMatchingWithMATLAB(argc, argv);
}

void compareMatchingWithMATLAB(int argc, char **argv) {
  int numFrames = 284;
  vector<SiftData> data(numFrames);

  for (int i = 0; i < numFrames; i++) {
    cerr << "Reading sift data for frame " << i + 1 << endl;
    std::ostringstream path;
    path << "../../../../../result/sift/sift";
    path << i + 1;
    ReadVLFeatSiftData(data[i], path.str().c_str());
  }

  for (int i = 0; i < numFrames - 1; i++) {
    vector<SiftMatch *> ours = MatchSiftData(data[i], data[i + 1], MatchSiftDistanceL2, 999, 0.6);

    std::ostringstream path;
    path << "../../../../../result/match/orig_match";
    path << i + 1;
    path << "_";
    path << i + 2;
    vector<SiftMatch *> theirs = ReadMATLABMatchDataBeforeRANSAC(path.str().c_str());

    cerr << ours.size() - theirs.size() << endl;
  }
}

void testMatchingL2(int argc, char **argv) {
  char *limgPath = argv[1];
  char *rimgPath = argv[2];

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

  // float total = 0.0f;
  // for (int i = 0; i < matches.size(); i++) {
  //   float sum = 0.0f;
  //   for (int j = 0; j < 128; j++) {
  //     sum += matches[i]->pt1->data[j] * matches[i]->pt2->data[j];
  //   }

  //   float diff = matches[i]->score - sum;
  //   // cerr << "Match " << i << " score " << 
  //   // matches[i]->score << ", " << sum << " " << diff << endl;
  //   total += abs(diff);
  // }

  // cerr << "Total error " << total << endl;

  vector<SiftMatch *> matchesCPU(numSift);
  for (int i = 0; i < numSift; i++) {
    float minScore = 999.0;
    float minScore2 = 999.0;
    int minIndex = 999;
    for (int j = 0; j < numSift; j++) {
      float score = 0.0;
      for (int k = 0; k < 128; k++) {
        score += (siftData1.h_data[i].data[k] - siftData2.h_data[j].data[k]) * (siftData1.h_data[i].data[k] - siftData2.h_data[j].data[k]);
      }

      if (score < minScore) {
        minScore2 = minScore;
        minScore = score;
        minIndex = j;
      } else if (score < minScore2) {
        minScore2 = score;
      }
    }

    SiftMatch *siftMatch = new SiftMatch();
    siftMatch->pt1 = &(siftData1.h_data[i]);
    siftMatch->pt2 = &(siftData2.h_data[minIndex]);
    siftMatch->score = minScore;
    siftMatch->ambiguity = minScore / (minScore2 + 1e-6);

    matchesCPU[i] = siftMatch;
  }

  float scoreSum = 0.0;
  float ambiguitySum = 0.0;
  for (int i = 0; i < numSift; i++) {
    scoreSum += abs(matches[i]->score - matchesCPU[i]->score);
    ambiguitySum += abs(matches[i]->ambiguity - matchesCPU[i]->ambiguity);
    // cerr << matches[i]->score - matchesCPU[i]->score << endl;
    // cerr << matches[i]->ambiguity - matchesCPU[i]->ambiguity << endl;
  }

  cerr << "Average score difference: " << scoreSum / numSift << endl;
  cerr << "Average ambiguity difference: " << ambiguitySum / numSift << endl;
  cerr << "Total score difference: " << scoreSum << endl;
  cerr << "Total ambiguity difference: " << ambiguitySum << endl;

  for (int i = 0; i < matches.size(); i++) {
    delete matches[i];
  }

  for (int i = 0; i < matchesCPU.size(); i++) {
    delete matchesCPU[i];
  }

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void testMatchingDotProduct(int argc, char **argv) {
  char *limgPath = argv[1];
  char *rimgPath = argv[2];

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

  vector<SiftMatch *> matches = MatchSiftData(siftData1, siftData2, MatchSiftDistanceDotProduct);

  float total = 0.0f;
  for (int i = 0; i < matches.size(); i++) {
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
      sum += matches[i]->pt1->data[j] * matches[i]->pt2->data[j];
    }

    float diff = matches[i]->score - sum;
    // cerr << "Match " << i << " score " << 
    // matches[i]->score << ", " << sum << " " << diff << endl;
    total += abs(diff);
  }

  cerr << "Total error " << total << endl;

  vector<SiftMatch *> matchesCPU(numSift);
  for (int i = 0; i < numSift; i++) {
    float maxScore = -1.0;
    float maxScore2 = -1.0;
    int maxIndex = -1;
    for (int j = 0; j < numSift; j++) {
      float score = 0.0;
      for (int k = 0; k < 128; k++) {
        score += siftData1.h_data[i].data[k] * siftData2.h_data[j].data[k];
      }

      if (score > maxScore) {
        maxScore2 = maxScore;
        maxScore = score;
        maxIndex = j;
      } else if (score > maxScore2) {
        maxScore2 = score;
      }
    }

    SiftMatch *siftMatch = new SiftMatch();
    siftMatch->pt1 = &(siftData1.h_data[i]);
    siftMatch->pt2 = &(siftData2.h_data[maxIndex]);
    siftMatch->score = maxScore;
    siftMatch->ambiguity = (1 - maxScore) / (1 - maxScore2 + 1e-6);

    matchesCPU[i] = siftMatch;
  }

  float scoreSum = 0.0;
  float ambiguitySum = 0.0;
  for (int i = 0; i < numSift; i++) {
    scoreSum += abs(matches[i]->score - matchesCPU[i]->score);
    ambiguitySum += abs(matches[i]->ambiguity - matchesCPU[i]->ambiguity);
    // cerr << matches[i]->score - matchesCPU[i]->score << endl;
    // cerr << matches[i]->ambiguity - matchesCPU[i]->ambiguity << endl;
  }

  cerr << "Average score difference: " << scoreSum / numSift << endl;
  cerr << "Average ambiguity difference: " << ambiguitySum / numSift << endl;
  cerr << "Total score difference: " << scoreSum << endl;
  cerr << "Total ambiguity difference: " << ambiguitySum << endl;

  for (int i = 0; i < matches.size(); i++) {
    delete matches[i];
  }

  for (int i = 0; i < matchesCPU.size(); i++) {
    delete matchesCPU[i];
  }

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void demo(int argc, char **argv) {
  if (argc < 4) {
    cout << "Usage: ./cuSIFT leftImagePath rightImagePath outDir numDevices" << endl;
    exit(1);
  }

  char *limgPath = argv[1];
  char *rimgPath = argv[2];
  char *outDir = argv[3];

  int devNum = 0;
  if (argc > 4)
    devNum = atoi(argv[4]);

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
  InitCuda(devNum);
  cuImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 0.0f;
  float thresh = 0.1f;
  InitSiftData(siftData1, 4096, true, true); 
  InitSiftData(siftData2, 4096, true, true);
  
  ExtractSift(siftData1, img1, 6, initBlur, thresh, 0.0f);
  ExtractSift(siftData2, img2, 6, initBlur, thresh, 0.0f);  

  // Match Sift features and find a homography
  MatchSiftData(siftData1, siftData2);
  float homography[9];
  int numMatches;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
  int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0);

  // Print out and store summary data
  PrintMatchData(siftData1, siftData2, img1);
#if 0
  PrintSiftData(siftData1);
  MatchAll(siftData1, siftData2, homography);
#endif
  cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << endl;
  cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numMatches/min(siftData1.numPts, siftData2.numPts) << "%" << endl;
  cv::imwrite(outDir, limg);

  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].coords2D[0] + homography[7]*sift1[i].coords2D[1] + homography[8];
      float dx = (homography[0]*sift1[i].coords2D[0] + homography[1]*sift1[i].coords2D[1] + homography[2]) / den - sift2[j].coords2D[0];
      float dy = (homography[3]*sift1[i].coords2D[0] + homography[4]*sift1[i].coords2D[1] + homography[5]) / den - sift2[j].coords2D[1];
      float err = dx*dx + dy*dy;
      if (err<100.0f)
	found = true;
      if (err<100.0f || j==sift1[i].match) {
	if (j==sift1[i].match && err<100.0f)
	  cout << " *";
	else if (j==sift1[i].match) 
	  cout << " -";
	else if (err<100.0f)
	  cout << " +";
	else
	  cout << "  ";
	cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << endl;
      }
    }
    cout << endl;
    if (found)
      numFound++;
  }
  cout << "Number of founds: " << numFound << endl;
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, cuImage &img)
{
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  cout << setprecision(3);
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (true || sift1[j].match_error<5) {
      float dx = sift2[k].coords2D[0] - sift1[j].coords2D[0];
      float dy = sift2[k].coords2D[1] - sift1[j].coords2D[1];
#if 1
      if (false && sift1[j].coords2D[0]>550 && sift1[j].coords2D[0]<600) {
	cout << "pos1=(" << (int)sift1[j].coords2D[0] << "," << (int)sift1[j].coords2D[1] << ") ";
	cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	cout << "scale=" << sift1[j].scale << "  ";
	cout << "error=" << (int)sift1[j].match_error << "  ";
	cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	cout << " delta=(" << (int)dx << "," << (int)dy << ")" << endl;
      }
#endif
#if 1
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].coords2D[0] + dx*l/len);
	int y = (int)(sift1[j].coords2D[1] + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }	
#endif
    }
#if 1
    int x = (int)(sift1[j].coords2D[0]+0.5);
    int y = (int)(sift1[j].coords2D[1]+0.5);
    int s = min(x, min(y, min(w-x-2, min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
#endif
  }
  cout << setprecision(6);
}


