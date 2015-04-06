#include "face_det.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

namespace masa_face_reg_rec {
using namespace std;
using namespace cv;

const string kDefaultCascadeName =
    "../data/haarcascades/haarcascade_frontalface_alt.xml";
const string kDefaultNestedCascadeName =
    "../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

bool FaceDetector::Init(string cascade_name, string nested_cascade_name) {
  if (cascade_name.empty()) {
    cascade_name = kDefaultCascadeName;
  }
  if (!cascade_.load(cascade_name)) {
    cout << "Failed to load cascade!\n";
    return false;
  }
  if (nested_cascade_name.empty()) {
    nested_cascade_name = kDefaultNestedCascadeName;
  }
  nested_cascade_.load(nested_cascade_name);
  return true;
}

vector<Rect> FaceDetector::Detect(Mat &img) {
  vector<Rect> faces;
  const static Scalar colors[] = {CV_RGB(0, 0, 255),   CV_RGB(0, 128, 255),
                                  CV_RGB(0, 255, 255), CV_RGB(0, 255, 0),
                                  CV_RGB(255, 128, 0), CV_RGB(255, 255, 0),
                                  CV_RGB(255, 0, 0),   CV_RGB(255, 0, 255)};
  Mat gray,
      small_img(round(img.rows / scale_), round(img.cols / scale_), CV_8UC1);

  cvtColor(img, gray, COLOR_BGR2GRAY);
  resize(gray, small_img, small_img.size(), 0, 0, INTER_LINEAR);
  equalizeHist(small_img, small_img);

  double t = (double)getTickCount();
  cascade_.detectMultiScale(small_img, faces, 1.1, 2,
                            0
                                //|CV_HAAR_FIND_BIGGEST_OBJECT
                                //|CV_HAAR_DO_ROUGH_SEARCH
                                |
                                CV_HAAR_SCALE_IMAGE,
                            Size(30, 30));
  t = (double)getTickCount() - t;
  // printf("detection time = %g ms\n",
  // t / ((double)cvGetTickFrequency() * 1000.));
  int i=0;
  for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
       r++, i++) {
    Point center;
    Scalar color = colors[i % 8];
    int radius;

    double aspect_ratio = (double)r->width / r->height;
    if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
      center.x = round((r->x + r->width * 0.5) * scale_);
      center.y = round((r->y + r->height * 0.5) * scale_);
      radius = round((r->width + r->height) * 0.25 * scale_);
      circle(img, center, radius, color, 3, 8, 0);
    } else
      rectangle(img, Point(round(r->x * scale_), round(r->y * scale_)),
                cvPoint(round((r->x + r->width - 1) * scale_),
                        round((r->y + r->height - 1) * scale_)),
                color, 3, 8, 0);
    if (nested_cascade_.empty())
      continue;
  }
  return faces;
}
}
