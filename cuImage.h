//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUIMAGE_H
#define CUIMAGE_H

class cuImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  cuImage();
  cuImage(int width, int height, float *h_data, bool download = true);
  ~cuImage();

  void AllocateWithHostMemory(int width, int height, float *h_data);
  void Allocate(int width, int height, int pitch, bool withHost, float *d_data = NULL, float *h_data = NULL);
  double DeviceToHost();
  double HostToDevice();
};

#endif // CUIMAGE_H
