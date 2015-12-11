#include "RGBD_utils.h"
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

void ransacfitRt(const cv::Mat refCoord, const cv::Mat movCoord, float* rigidtransform, 
 int* numMatches, int numLoops, float thresh)
{
  cv::Mat coord(refCoord.size().height, refCoord.size().width+movCoord.size().width, CV_32FC1);
  cv::Mat left(coord, cv::Rect(0, 0, refCoord.size().width, refCoord.size().height));
  refCoord.copyTo(left);
  cv::Mat right(coord, cv::Rect(refCoord.size().width, 0, movCoord.size().width, movCoord.size().height));
  movCoord.copyTo(right);
  float * h_coord = (float*)coord.data;
  //writeMatToFile(coord, "h_coord.txt");
  int numValid = refCoord.size().height;
  
  int randSize = 4*sizeof(int)*numLoops;
  int* h_randPts = (int*)malloc(randSize);
  
  // generate random samples for each loop
  for (int i=0;i<numLoops;i++) {
    int p1 = rand() % numValid;
    int p2 = rand() % numValid;
    int p3 = rand() % numValid;
    while (p2==p1) p2 = rand() % numValid;
    while (p3==p1 || p3==p2) p3 = rand() % numValid;
    h_randPts[i+0*numLoops] = p1;
    h_randPts[i+1*numLoops] = p2;
    h_randPts[i+2*numLoops] = p3;
  }

  int h_count =-1;
  float thresh2 = thresh*thresh;

#ifdef CPURANSAC
  float h_RT[12];
  int maxIndex = -1;
  int maxCount = -1;
  for(int idx= 0;idx<numLoops;idx++){

    estimateRigidTransform(h_coord, h_randPts, idx, numLoops, h_RT);
    TestRigidTransform(h_coord, h_RT, &h_count, numValid, thresh2);

    if (h_count>maxCount){
      maxCount = h_count;
      for (int i = 0;i<12;i++){
        rigidtransform[i] = h_RT[i];
      }
    }
  }

  numMatches[0] = maxCount;

#endif

  // gpu ransac;
  gpuRANSACfindRT(h_coord,h_randPts,rigidtransform,numMatches,numLoops,numValid,thresh2);
#ifdef VERBOSE
  cout << endl;
  cout << "RANSAC Fit Rt" << endl;

  for (int i = 0; i < 12; i++) {
    fprintf(stderr, "%0.4f ", rigidtransform[i]);
    if ((i + 1) % 4 == 0) cout << endl;
  }
  cout << endl;
  printf("Num loops: %d\n", numLoops);
  printf("Threshold %0.4f\n", thresh);

  printf("numofMatch = %d \n",numMatches[0]);
#endif
  // printf("rigidtransform\n");
  // for (int jj =0; jj<3;jj++){
  //     printf("%f,%f,%f,%f\n",rigidtransform[0+jj*4],rigidtransform[1+4*jj],rigidtransform[2+4*jj],rigidtransform[3+4*jj]);
  // }

  free(h_randPts);

  return;
}

unsigned int uchar2uint(unsigned char* in) {
  return (((unsigned int)(in[0])) << 16) + (((unsigned int)(in[1])) << 8) + ((unsigned int)(in[2]));
}

void uint2uchar(unsigned int in, unsigned char* out) {
  out[0] = (in & 0x00ff0000) >> 16;
  out[1] = (in & 0x0000ff00) >> 8;
  out[2] = in & 0x000000ff;
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
  //std::cout << std::setprecision(6);
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