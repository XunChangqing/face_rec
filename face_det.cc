#include "face_det.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

namespace masa_face_reg_rec {
using namespace std;
using namespace cv;

bool FaceDetector::Init(string cascade_name, string nested_cascade_name) {
  if (!cascade_.load(cascade_name)) {
    cout << "Failed to load cascade!\n";
    return false;
  }
  nested_cascade_.load(nested_cascade_name);
  return true;
}

vector<Rect> FaceDetector::Detect(Mat &img) {
  int i = 0;
  double t = 0;
  vector<Rect> faces;
  const static Scalar colors[] = {CV_RGB(0, 0, 255),   CV_RGB(0, 128, 255),
                                  CV_RGB(0, 255, 255), CV_RGB(0, 255, 0),
                                  CV_RGB(255, 128, 0), CV_RGB(255, 255, 0),
                                  CV_RGB(255, 0, 0),   CV_RGB(255, 0, 255)};
  Mat gray,
      smallImg(cvRound(img.rows / scale_), cvRound(img.cols / scale_), CV_8UC1);

  cvtColor(img, gray, CV_BGR2GRAY);
  resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
  equalizeHist(smallImg, smallImg);

  t = (double)cvGetTickCount();
  cascade_.detectMultiScale(smallImg, faces, 1.1, 2,
                            0
                                //|CV_HAAR_FIND_BIGGEST_OBJECT
                                //|CV_HAAR_DO_ROUGH_SEARCH
                                |
                                CV_HAAR_SCALE_IMAGE,
                            Size(30, 30));
  // if (tryflip) {
  // flip(smallImg, smallImg, 1);
  // cascade.detectMultiScale(smallImg, faces2, 1.1, 2,
  // 0
  ////|CV_HAAR_FIND_BIGGEST_OBJECT
  ////|CV_HAAR_DO_ROUGH_SEARCH
  //|
  // CV_HAAR_SCALE_IMAGE,
  // Size(30, 30));
  // for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end();
  // r++) {
  // faces.push_back(
  // Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
  //}
  //}
  t = (double)cvGetTickCount() - t;
  // printf("detection time = %g ms\n",
  // t / ((double)cvGetTickFrequency() * 1000.));
  for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
       r++, i++) {
    // Mat smallImgROI;
    // vector<Rect> nestedObjects;
    Point center;
    Scalar color = colors[i % 8];
    int radius;

    double aspect_ratio = (double)r->width / r->height;
    if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
      center.x = cvRound((r->x + r->width * 0.5) * scale_);
      center.y = cvRound((r->y + r->height * 0.5) * scale_);
      radius = cvRound((r->width + r->height) * 0.25 * scale_);
      circle(img, center, radius, color, 3, 8, 0);
    } else
      rectangle(img, cvPoint(cvRound(r->x * scale_), cvRound(r->y * scale_)),
                cvPoint(cvRound((r->x + r->width - 1) * scale_),
                        cvRound((r->y + r->height - 1) * scale_)),
                color, 3, 8, 0);
    if (nested_cascade_.empty())
      continue;
    // smallImgROI = smallImg(*r);
    // nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2,
    // 0
    ////|CV_HAAR_FIND_BIGGEST_OBJECT
    ////|CV_HAAR_DO_ROUGH_SEARCH
    ////|CV_HAAR_DO_CANNY_PRUNING
    //|
    // CV_HAAR_SCALE_IMAGE,
    // Size(30, 30));
    // for (vector<Rect>::const_iterator nr = nestedObjects.begin();
    // nr != nestedObjects.end(); nr++) {
    // center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
    // center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
    // radius = cvRound((nr->width + nr->height) * 0.25 * scale);
    // circle(img, center, radius, color, 3, 8, 0);
    //}
  }
  //imshow("result", img);
  return faces;
}
}
