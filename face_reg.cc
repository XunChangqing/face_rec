#include "face_det.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace masa_face_reg_rec;

static void help() {
  cout << "\nThis program demonstrates the cascade recognizer. Now you can use "
          "Haar or LBP features.\n"
          "This classifier can recognize many kinds of rigid objects, once the "
          "appropriate classifier is trained.\n"
          "It's most known use is for faces.\n"
          "Usage:\n"
          "./facedetect [--cascade=<cascade_path> this is the primary trained "
          "classifier such as frontal face]\n"
          "   [--nested-cascade[=nested_cascade_path this an optional "
          "secondary classifier such as eyes]]\n"
          "   [--scale=<image scale greater or equal to 1, try 1.3 for "
          "example>]\n"
          "   [--try-flip]\n"
          "   [filename|camera_index]\n\n"
          "see facedetect.cmd for one call:\n"
          "./facedetect "
          "--cascade=\"../../data/haarcascades/"
          "haarcascade_frontalface_alt.xml\" "
          "--nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" "
          "--scale=1.3\n\n"
          "During execution:\n\tHit any key to quit.\n"
          "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

string cascade_name = "../data/haarcascades/haarcascade_frontalface_alt.xml";
string nested_cascade_name =
    "../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

const int kFaceWidth = 380;
const int kFaceHeight = 380;

int main(int argc, const char **argv) {
  VideoCapture capture;
  Mat frame, frameCopy, image;
  const string scale_opt = "--scale=";
  size_t scale_opt_len = scale_opt.length();
  const string cascadeOpt = "--cascade=";
  size_t cascade_opt_len = cascadeOpt.length();
  const string nested_cascade_opt = "--nested-cascade";
  size_t nested_cascade_opt_len = nested_cascade_opt.length();
  // const string tryFlipOpt = "--try-flip";
  // size_t tryFlipOptLen = tryFlipOpt.length();
  const string input_opt = "--input=";
  size_t input_opt_len = input_opt.length();
  const string dir_opt = "--dir=";
  size_t dir_opt_len = dir_opt.length();
  string input_name;
  string dir_name;
  string person_reg_name;
  // bool tryflip = false;

  help();

  CascadeClassifier cascade, nested_cascade;
  double scale = 2.0;

  for (int i = 1; i < argc; i++) {
    cout << "Processing " << i << " " << argv[i] << endl;
    if (cascadeOpt.compare(0, cascade_opt_len, argv[i], cascade_opt_len) == 0) {
      cascade_name.assign(argv[i] + cascade_opt_len);
      cout << "  from which we have cascade_name= " << cascade_name << endl;
    } else if (nested_cascade_opt.compare(0, nested_cascade_opt_len, argv[i],
                                          nested_cascade_opt_len) == 0) {
      if (argv[i][nested_cascade_opt.length()] == '=')
        nested_cascade_name.assign(argv[i] + nested_cascade_opt.length() + 1);
      if (!nested_cascade.load(nested_cascade_name))
        cerr << "WARNING: Could not load classifier cascade for nested objects"
             << endl;
    } else if (scale_opt.compare(0, scale_opt_len, argv[i], scale_opt_len) ==
               0) {
      if (!sscanf(argv[i] + scale_opt.length(), "%lf", &scale) || scale < 1)
        scale = 1;
      cout << " from which we read scale = " << scale << endl;
    } else if (input_opt.compare(0, input_opt_len, argv[i], input_opt_len) ==
               0) {
      input_name.assign(argv[i] + input_opt_len);
    } else if (dir_opt.compare(0, dir_opt_len, argv[i], dir_opt_len) == 0) {
      dir_name.assign(argv[i] + dir_opt_len);
    }
    // else if (tryFlipOpt.compare(0, tryFlipOptLen, argv[i], tryFlipOptLen) ==
    // 0) {
    // tryflip = true;
    // cout << " will try to flip image horizontally to detect assymetric "
    //"objects\n";
    //}
    else if (argv[i][0] == '-') {
      cerr << "WARNING: Unknown option " << argv[i] << endl;
    } else
      person_reg_name.assign(argv[i]);
  }

  if (person_reg_name.empty()) {
    cerr << "Person name is empty!\n";
    exit(1);
  }

  if (dir_name.empty()) {
    dir_name.assign(".");
  }
  string person_dir_name = dir_name + "/" + person_reg_name + "/";
  boost::filesystem::path person_dir(person_dir_name);
  boost::filesystem::create_directories(person_dir);

  if (input_name.empty() ||
      (isdigit(input_name.c_str()[0]) && input_name.c_str()[1] == '\0')) {
    capture.open(input_name.empty() ? 0 : input_name.c_str()[0] - '0');
  } else {
    capture.open(input_name);
  }
  if (!capture.isOpened()) {
    cerr << "Failed to open video capture!\n";
    exit(1);
  }

  FaceDetector face_det(scale);
  if (!face_det.Init(cascade_name, nested_cascade_name)) {
    cerr << "Cannot init face detector!\n";
    exit(1);
  }

  namedWindow("result", 1);

  cout << "In capture ..." << endl;
  int face_idx = 0;
  for (;;) {
    capture >> frame;
    if (frame.empty())
      break;

    Mat frame_copy = frame.clone();
    vector<Rect> faces = face_det.Detect(frame_copy);

    int ret = waitKey(1);
    if (ret == 27)
      break;
    else if (faces.size() == 1) {
      Rect real_face_rect(faces[0].x * scale, faces[0].y * scale,
                          faces[0].width * scale, faces[0].height * scale);
      Mat face_mat = frame(real_face_rect);
      imshow("face", face_mat);
      if (ret == 32) {
        cout << "Store face " << face_idx << endl;
	Mat resized_face_mat;
        resize(face_mat, resized_face_mat, Size(kFaceWidth, kFaceHeight));
        imwrite(person_dir_name + to_string(face_idx++) + ".jpg",
                resized_face_mat);
      }
    }
  }

  // destroyWindow("result");
  return 0;
}
