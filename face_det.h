#ifndef FACE_REG_REC_FACE_DET_H_
#define FACE_REG_REC_FACE_DET_H_
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <string>
#include <vector>

namespace masa_face_reg_rec {
class FaceDetector {
public:
  explicit FaceDetector(double scale) : scale_(scale) {}
  bool Init(std::string cascade_name, std::string nested_cascade_name);
  std::vector<cv::Rect> Detect(cv::Mat &img);
  // std::vector<cv::Rect> get_faces(){return faces_;};

private:
  double scale_;
  cv::CascadeClassifier cascade_;
  cv::CascadeClassifier nested_cascade_;
  std::vector<cv::Rect> faces_;

  FaceDetector(const FaceDetector&);
  void operator=(const FaceDetector&);
};
}

#endif
