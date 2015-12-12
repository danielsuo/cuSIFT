#include "debug.h"
////////////////////////////////////////////////////////////////////////////////

void writeMatToFile(cv::Mat& m, const char* filename)
{
  ofstream fout(filename);

  double sum = 0;

  if(!fout) {
    cout<<"File Not Opened"<<endl;  return;
  }

  for(int i = 0; i < m.rows; i++) {
    for(int j = 0; j < m.cols; j++) {
      fout << m.at<float>(i, j) << "\t";
      sum += m.at<float>(i, j);
    }
    fout << endl;
  }

  fprintf(stderr, "Sum of matrix: %f\n", sum);

  fout.close();
}

cv::Mat PrintMatchData(SiftData &siftData1, SiftData &siftData2, cv::Mat limg, cv::Mat rimg)
{
  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  cv::Mat im3(limg.size().height, limg.size().width + rimg.size().width, CV_32FC1);
  cv::Mat left(im3, cv::Rect(0, 0, limg.size().width, limg.size().height));
  limg.copyTo(left);
  cv::Mat right(im3, cv::Rect(limg.size().width, 0, rimg.size().width, rimg.size().height));
  rimg.copyTo(right);

  int w = limg.size().width + rimg.size().width;
  for (int j = 0; j < numPts; j++) {
    if (sift1[j].valid == 1) {
      float dx = sift1[j].match_xpos + limg.size().width - sift1[j].xpos;
      float dy = sift1[j].match_ypos - sift1[j].ypos;
      int len = (int)(fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l = 0; l < len; l++) {
        int x = (int)(sift1[j].xpos + dx * l / len);
        int y = (int)(sift1[j].ypos + dy * l / len);
        im3.at<float>(y, x) = 255.0f;
      }
    }
  }

  return im3;
}

void PrintMatchSiftData(SiftData &siftData1, const char* filename, int imgw) {
  ofstream fout(filename);
  if (!fout)
  {
    cout << "File Not Opened" << endl;  return;
  }
  SiftPoint *sift1 = siftData1.h_data;
  for (int i = 0; i < siftData1.numPts; i++)
  {
    if (sift1[i].valid) {
      int ind  = ((int)sift1[i].xpos + (int)sift1[i].ypos * imgw);
      int ind2 = ((int)sift1[i].match_xpos + (int)sift1[i].match_ypos * imgw);

      fout << sift1[i].xpos << "\t" << sift1[i].ypos << "\t";
      fout << sift1[i].match_xpos << "\t" << sift1[i].match_ypos << "\t";
      fout << ind << "\t" << ind2 << "\t";
      fout << endl;
    }
  }

  fout.close();
}

/* Assumes output was created by vl_sift_tofile.m
 * Binary format
 * - uint32_t numPts: number of sift points that will follow
 * - float points: 4 x numPts that contains x, y, scale, and orientation 
 *   (in radians)
 * - float descriptors: 128 x numPts that contain descriptors (0.0 to 1.0)
*/
void ReadVLFeatSiftData(SiftData &siftData, const char *filename) {
  fprintf(stderr, "Reading vlfeat data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  // First, read number of SIFT points in the VLFeat SIFT data file
  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  // Next, grab matrix containing x, y, scale, orientation (in radians)
  float *points = new float[4 * numPts];
  fread((void *)points, sizeof(float), 4 * numPts, fp);

  // Finally, get SIFT descriptor arrays
  float *descriptors = new float[128 * numPts];
  fread((void *)descriptors, sizeof(float), 128 * numPts, fp);

  // Finish reading
  fclose(fp);

  SiftPoint *h_data = (SiftPoint *)calloc(numPts, sizeof(SiftPoint));

  for (int i = 0; i < numPts; i++) {
    h_data[i].xpos = points[i * 4];
    h_data[i].ypos = points[i * 4 + 1];
    h_data[i].scale = points[i * 4 + 2];
    h_data[i].orientation = points[i * 4 + 3];

    memcpy(h_data[i].data, descriptors + i * 128, sizeof(float) * 128);
  }

  AddSiftData(siftData, h_data, numPts);

  free(points);
  free(descriptors);
  free(h_data);
}

void ReadMATLABMatchData(cv::Mat &curr_match, cv::Mat &next_match, const char *filename) {
  fprintf(stderr, "Reading MATLAB match data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  // First, read number of matched points
  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  // Next, grab matrix containing world coordinates from current frame and
  // world coordinates from next frame (x1, y1, z1, x2, y2, z2)
  float *matchedPoints = new float[6 * numPts];
  fread((void *)matchedPoints, sizeof(float), 6 * numPts, fp);

  // Resize the match matrices
  curr_match.resize(numPts);
  next_match.resize(numPts);

  fprintf(stderr, "Number of matches %d\n", curr_match.rows);
  // Store match data in curr_match and next_match, which are 3xnumPts matrices
  for (int i = 0; i < numPts; i++) {
    curr_match.at<float>(i, 0) = matchedPoints[i * 6 + 0];
    curr_match.at<float>(i, 1) = matchedPoints[i * 6 + 1];
    curr_match.at<float>(i, 2) = matchedPoints[i * 6 + 2];
    next_match.at<float>(i, 0) = matchedPoints[i * 6 + 3];
    next_match.at<float>(i, 1) = matchedPoints[i * 6 + 4];
    next_match.at<float>(i, 2) = matchedPoints[i * 6 + 5];

    fprintf(stderr, "Matches %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n",
      curr_match.at<float>(i, 0),
      curr_match.at<float>(i, 1),
      curr_match.at<float>(i, 2),
      next_match.at<float>(i, 0),
      next_match.at<float>(i, 1),
      next_match.at<float>(i, 2)
      );
  }

  free(matchedPoints);
}

void ReadMATLABRt(float *Rt_relative, const char *filename) {
  fprintf(stderr, "Reading MATLAB Rt data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");
  fread((void *)Rt_relative, sizeof(float), 12, fp);
  fclose(fp);

  fprintf(stderr, "MATLAB Rt: ");
  for (int i = 0; i < 12; i++) {
    fprintf(stderr, "%0.4f ", Rt_relative[i]);
  }
  fprintf(stderr, "\n");
}