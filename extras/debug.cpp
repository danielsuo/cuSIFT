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

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data.maxPts));
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].coords2D[0]);
    printf("ypos         = %.2f\n", h_data[i].coords2D[1]);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float*)&h_data[i].data;
    for (int j = 0; j < 8; j++) {
      if (j == 0) {
        printf("data = ");
      }
      else {
        printf("       ");
      }
      for (int k = 0; k<16; k++) {
        if (siftData[j * 16 + k] < 0.01) {
          printf(" .   ");
        }
        else {
          printf("%.2f ", siftData[j * 16 + k]);
        }
      }
      
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
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
    float dx = sift1[j].match_xpos + limg.size().width - sift1[j].coords2D[0];
    float dy = sift1[j].match_ypos - sift1[j].coords2D[1];
    int len = (int)(fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
    for (int l = 0; l < len; l++) {
      int x = (int)(sift1[j].coords2D[0] + dx * l / len);
      int y = (int)(sift1[j].coords2D[1] + dy * l / len);
      im3.at<float>(y, x) = 255.0f;
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
    int ind  = ((int)sift1[i].coords2D[0] + (int)sift1[i].coords2D[1] * imgw);
    int ind2 = ((int)sift1[i].match_xpos + (int)sift1[i].match_ypos * imgw);

    fout << sift1[i].coords2D[0] << "\t" << sift1[i].coords2D[1] << "\t";
    fout << sift1[i].match_xpos << "\t" << sift1[i].match_ypos << "\t";
    fout << ind << "\t" << ind2 << "\t";
    fout << endl;
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
  fprintf(stderr, "Reading vlfeat data from %s", filename);

  // InitSiftData(siftData, 1024, true, true);

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

  fprintf(stderr, " ... and got %d points\n", numPts);

  for (int i = 0; i < numPts; i++) {
    h_data[i].coords2D[0] = points[i * 4];
    h_data[i].coords2D[1] = points[i * 4 + 1];
    h_data[i].scale = points[i * 4 + 2];
    h_data[i].orientation = points[i * 4 + 3];

    memcpy(h_data[i].data, descriptors + i * 128, sizeof(float) * 128);
  }

  AddSiftData(siftData, h_data, numPts);

  free(points);
  free(descriptors);
  free(h_data);
}

int ReadMATLABMatchIndices(const char *indices_filename, uint32_t *indices_i, uint32_t *indices_j) {
  fprintf(stderr, "Reading match indices data from %s\n", indices_filename);
  FILE *fp = fopen(indices_filename, "rb");

  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  if (indices_i != NULL && indices_j != NULL) {
    fread((void *)indices_i, sizeof(uint32_t), numPts, fp);
    fread((void *)indices_j, sizeof(uint32_t), numPts, fp);
  }
  fclose(fp);

  return numPts;
}

vector<float *> ReadVLFeatSiftDataAsFloatArray(const char *filename) {
  fprintf(stderr, "Reading vlfeat sift data as double array from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);
  vector<float *> siftPoints;

  for (int i = 0; i < numPts; i++) {
    float siftPoint[128];
    fread((void *)siftPoint, sizeof(float), 128, fp);
    siftPoints.push_back(siftPoint);
  }

  fclose(fp);

  return siftPoints;
}

void ReadMATLABMatchData(cv::Mat &curr_match, cv::Mat &next_match, const char *filename) {
  fprintf(stderr, "Reading MATLAB match data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  // First, read number of matched points
  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  // Next, grab matrix containing world coordinates from current frame and
  // world coordinates from next frame (x1, y1, z1, x2, y2, z2)
  double *matchedPoints = new double[6 * numPts];
  fread((void *)matchedPoints, sizeof(double), 6 * numPts, fp);

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

vector<SiftMatch *> ReadMATLABMatchData(const char *filename) {
  fprintf(stderr, "Reading MATLAB match data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  vector<SiftMatch *> matches;

  for (int i = 0; i < numPts; i++) {
    SiftMatch *match = new SiftMatch();
    SiftPoint *pt1 = new SiftPoint();
    SiftPoint *pt2 = new SiftPoint();

    double coords1[3];
    double coords2[3];

    fread((void *)coords1, sizeof(double), 3, fp);
    fread((void *)coords2, sizeof(double), 3, fp);

    pt1->coords3D[0] = coords1[0];
    pt1->coords3D[1] = coords1[1];
    pt1->coords3D[2] = coords1[2];
    pt2->coords3D[0] = coords2[0];
    pt2->coords3D[1] = coords2[1];
    pt2->coords3D[2] = coords2[2];

    match->pt1 = pt1;
    match->pt2 = pt2;

    matches.push_back(match);
  }

  return(matches);
}

// Read MATLAB before RANSAC
vector<SiftMatch *> ReadMATLABMatchDataBeforeRANSAC(const char *filename) {
  fprintf(stderr, "Reading MATLAB match data before ransac from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  uint32_t numPts;
  fread((void *)&numPts, sizeof(uint32_t), 1, fp);

  int matchPointsID_i[numPts];
  fread((void *)matchPointsID_i, sizeof(uint32_t), numPts, fp);

  int matchPointsID_j[numPts];
  fread((void *)matchPointsID_j, sizeof(uint32_t), numPts, fp);

  float SIFTdes_i[numPts * 128];
  fread((void *)SIFTdes_i, sizeof(float), numPts * 128, fp);

  float SIFTdes_j[numPts * 128];
  fread((void *)SIFTdes_j, sizeof(float), numPts * 128, fp);

  vector<SiftMatch *> matches;

  for (int i = 0; i < numPts; i++) {
    SiftMatch *match = new SiftMatch();
    SiftPoint *pt1 = new SiftPoint();
    SiftPoint *pt2 = new SiftPoint();

    memcpy(pt1->data, SIFTdes_i + i * 128, sizeof(float) * 128);
    memcpy(pt2->data, SIFTdes_j + i * 128, sizeof(float) * 128);

    matches.push_back(match);
  }

  return matches;
}

vector<SiftMatch *> ReadMATLABRANSAC(const char *filename, vector<int> &indices, float *Rt) {

  fprintf(stderr, "Reading MATLAB RANSAC data produced using DEBUG_ransactfitRt.m from %s\n", filename);

  FILE *fp = fopen(filename, "rb");

  uint32_t numMatches;
  fread((void *)&numMatches, sizeof(uint32_t), 1, fp);
  vector<SiftMatch *> matches(numMatches);

  uint32_t numLoops;
  fread((void *)&numLoops, sizeof(uint32_t), 1, fp);

  fprintf(stderr, "Read %d matches and %d loop indices\n", numMatches, numLoops);

  float *coords3D_i = new float[numMatches * 3];
  fread((void *)coords3D_i, sizeof(float), numMatches * 3, fp);

  float *coords3D_j = new float[numMatches * 3];
  fread((void *)coords3D_j, sizeof(float), numMatches * 3, fp);

  for (int i = 0; i < numMatches; i++) {
    SiftMatch *match = new SiftMatch();
    SiftPoint *pt1 = new SiftPoint();
    SiftPoint *pt2 = new SiftPoint();

    memcpy(pt1->coords3D, coords3D_i + i * 3, sizeof(float) * 3);
    memcpy(pt2->coords3D, coords3D_j + i * 3, sizeof(float) * 3);

    match->pt1 = pt1;
    match->pt2 = pt2;

    matches[i] = match;
  }

  for (int i = 0; i < numLoops * 3; i++) {
    int index;
    fread((void *)&index, sizeof(int), 1, fp);
    indices.push_back(index - 1);
  }

  fread((void *)Rt, sizeof(float), 12, fp);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      fprintf(stderr, "%0.4f ", Rt[i * 4 + j]);
    }
    fprintf(stderr, "\n");
  }

  delete [] coords3D_i;
  delete [] coords3D_j;

  return matches;
}

vector<int> ReadMATLABIndices(const char *filename) {
  fprintf(stderr, "Reading MATLAB indices data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");
  uint32_t nPairs;
  fread((void *)&nPairs, sizeof(uint32_t), 1, fp);

  vector<int> results;
  for (int i = 0; i < nPairs; i++) {
    uint32_t tmp;
    fread((void *)&tmp, sizeof(uint32_t), 1, fp);
    results.push_back(tmp - 1);

    fread((void *)&tmp, sizeof(uint32_t), 1, fp);
    results.push_back(tmp - 1);
  }

  return results;
}

void ReadMATLABRt(double *Rt_relative, const char *filename) {
  fprintf(stderr, "Reading MATLAB Rt data from %s\n", filename);

  FILE *fp = fopen(filename, "rb");
  fread((void *)Rt_relative, sizeof(double), 12, fp);
  fclose(fp);

  fprintf(stderr, "MATLAB Rt: ");
  for (int i = 0; i < 12; i++) {
    fprintf(stderr, "%0.4f ", Rt_relative[i]);
  }
  fprintf(stderr, "\n");
}

// Modifying the original AddSiftData to add data from CPU memory, rather than
// from GPU memory. Also, ignore subsampling for now.
//
// Original function:
// void AddSiftData(SiftData &data, float *d_sift, float *d_desc, int numPts, int maxPts, float subsampling)
void AddSiftData(SiftData &data, SiftPoint *h_data, int numPts) {
  // Compute new total number of points once we add new points
  int newNum = data.numPts + numPts;

  // If we haven't allocated enough memory for all of the points, double the
  // memory
  if (data.maxPts < newNum) {

    // Get new amount of memory to allocate
    int newMaxNum = 2 * data.maxPts;
    while (newNum > newMaxNum)
      newMaxNum *= 2;

    // If we have host data, allocate new memory, copy over, and free old memory
    if (data.h_data != NULL) {
      SiftPoint *tmp = (SiftPoint *)malloc(sizeof(SiftPoint) * newMaxNum);
      memcpy(tmp, data.h_data, sizeof(SiftPoint) * data.numPts);
      free(data.h_data);
      data.h_data = tmp;
    }

    // If we have device data, allocate new memory, copy over, and free old memory
    if (data.d_data != NULL) {
      SiftPoint *d_data = NULL;
      safeCall(cudaMalloc((void**)&d_data, sizeof(SiftPoint) * newMaxNum));
      safeCall(cudaMemcpy(d_data, data.d_data, sizeof(SiftPoint) * data.numPts, cudaMemcpyDeviceToDevice));
      safeCall(cudaFree(data.d_data));
      data.d_data = d_data;
    }
    data.maxPts = newMaxNum;
  }

  if (data.h_data != NULL) {
    memcpy(data.h_data + data.numPts, h_data, sizeof(SiftPoint) * numPts);
  }

  if (data.d_data != NULL) {
    safeCall(cudaMemcpy(data.d_data + data.numPts, h_data, sizeof(SiftPoint) * numPts, cudaMemcpyHostToDevice));
  }

  data.numPts = newNum;
}