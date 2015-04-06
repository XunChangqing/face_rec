#include "face_det.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace masa_face_reg_rec;
namespace fs = boost::filesystem;

double scale = 2.0;
string cascade_name = "../data/haarcascades/haarcascade_frontalface_alt.xml";
string nested_cascade_name =
    "../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

void read_images(const string &dir_name, vector<Mat> &images,
                 vector<string> &label_names, vector<int> &labels) {
  fs::path person_dir(dir_name);
  int lid = 0;
  for (fs::directory_iterator person_iter(person_dir);
       person_iter != fs::directory_iterator(); person_iter++) {
    if (fs::is_directory(*person_iter)) {
      for (fs::directory_iterator face_iter(*person_iter);
           face_iter != fs::directory_iterator(); face_iter++) {
        if (fs::is_regular_file(*face_iter)) {
          images.push_back(
              imread(face_iter->path().string(), CV_LOAD_IMAGE_GRAYSCALE));
          // image_names.push_back((face_iter->path().string()));
          labels.push_back(lid);
        }
      }
      label_names.push_back(person_iter->path().filename().string());
      ++lid;
    }
  }
}

int main(int argc, const char *argv[]) {
  // Check for valid command line arguments, print usage
  // if no arguments were given.
  if (argc != 2) {
    cout << "usage: " << argv[0] << " dir" << endl;
    exit(1);
  }
  string person_dir = string(argv[1]);
  // Get the path to your CSV.
  // string fn_csv = string(argv[1]);
  // These vectors hold the images and corresponding labels.
  vector<Mat> images;
  // vector<string> image_names;
  vector<string> label_names;
  vector<int> labels;
  read_images(person_dir, images, label_names, labels);
  for (vector<string>::iterator liter = label_names.begin();
       liter != label_names.end(); liter++) {
    cout << *liter << endl;
  }
  // Quit if there are not enough images for this demo.
  if (images.size() <= 1) {
    string error_message = "This demo needs at least 2 images to work. Please "
                           "add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  // Get the height from the first image. We'll need this
  // later in code to reshape the images to their original
  // size:
  int height = images[0].rows;
  cout << "height: " << height << endl;
  // The following lines simply get the last images from
  // your dataset and remove it from the vector. This is
  // done, so that the training data (which we learn the
  // cv::FaceRecognizer on) and the test data we test
  // the model with, do not overlap.
  //cout << images.size() << " " << labels.size() << endl;
  //Mat testSample = images[images.size() - 1];
  //int testLabel = labels[labels.size() - 1];
  //images.pop_back();
  //labels.pop_back();
  // The following lines create an Eigenfaces model for
  // face recognition and train it with the images and
  // labels read from the given CSV file.
  // This here is a full PCA, if you just want to keep
  // 10 principal components (read Eigenfaces), then call
  // the factory method like this:
  //
  //      cv::createEigenFaceRecognizer(10);
  //
  // If you want to create a FaceRecognizer with a
  // confidennce threshold, call it with:
  //
  //      cv::createEigenFaceRecognizer(10, 123.0);
  //
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  cout << images.size() << " " << labels.size() << endl;
  model->train(images, labels);
  //cout << "After train!" << endl;
  // The following line predicts the label of a given
  // test image:
  // int predictedLabel = model->predict(testSample);
  //
  // To get the confidence of a prediction call the model with:
  //
  //int predictedLabel = -1;
  //double confidence = 0.0;
  //model->predict(testSample, predictedLabel, confidence);
  //cout << "After predict!" << endl;

  //string result_message =
      //format("Predicted class = %d / Actual class = %d / Confidence = %f.",
             //predictedLabel, testLabel, confidence);
  //cout << result_message << endl;
  //cout << label_names[predictedLabel] << endl;

  FaceDetector face_det(scale);
  if (!face_det.Init(cascade_name, nested_cascade_name)) {
    cerr << "Cannot init face detector!\n";
    exit(1);
  }
  VideoCapture capture(0);

  Mat frame;
  for (;;) {
    capture >> frame;
    if (frame.empty())
      break;

    Mat frame_copy = frame.clone();
    vector<Rect> faces = face_det.Detect(frame_copy);

    Scalar color = CV_RGB(255, 0, 0);
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
         r++) {
      Rect real_face_rect(r->x * scale, r->y * scale, r->width * scale,
                          r->height * scale);
      Mat face_mat = frame(real_face_rect);
      Mat resized_face_mat;
      resize(face_mat, resized_face_mat, images[0].size());
      cvtColor(resized_face_mat, resized_face_mat, CV_BGR2GRAY);

      int predicted_label = -1;
      double confidence = 0.0;
      model->predict(resized_face_mat, predicted_label, confidence);

      Point center;
      center.x = cvRound((r->x + r->width * 0.5) * scale);
      center.y = cvRound((r->y + r->height * 0.5) * scale);
      if (predicted_label >= 0 && predicted_label < label_names.size()) {
        putText(frame_copy, label_names[predicted_label], center,
                FONT_HERSHEY_SIMPLEX, 2, color);
      }
    }

    imshow("faces", frame_copy);
    if (waitKey(1) == 27)
      break;
  }
  // Sometimes you'll need to get/set internal model data,
  // which isn't exposed by the public cv::FaceRecognizer.
  // Since each cv::FaceRecognizer is derived from a
  // cv::Algorithm, you can query the data.
  //
  // First we'll use it to set the threshold of the FaceRecognizer
  // to 0.0 without retraining the model. This can be useful if
  // you are evaluating the model:
  //
  // model->set("threshold", 0.0);
  //// Now the threshold of this model is set to 0.0. A prediction
  //// now returns -1, as it's impossible to have a distance below
  //// it
  // predictedLabel = model->predict(testSample);
  // cout << "Predicted class = " << predictedLabel << endl;
  //// Here is how to get the eigenvalues of this Eigenfaces model:
  // Mat eigenvalues = model->getMat("eigenvalues");
  //// And we can do the same to display the Eigenvectors (read Eigenfaces):
  // Mat W = model->getMat("eigenvectors");
  //// From this we will display the (at most) first 10 Eigenfaces:
  // for (int i = 0; i < min(10, W.cols); i++) {
  // string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
  // cout << msg << endl;
  //// get eigenvector #i
  // Mat ev = W.col(i).clone();
  //// Reshape to original size & normalize to [0...255] for imshow.
  // Mat grayscale = toGrayscale(ev.reshape(1, height));
  //// Show the image & apply a Jet colormap for better sensing.
  // Mat cgrayscale;
  // applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
  // imshow(format("%d", i), cgrayscale);
  //}
  // waitKey(0);

  return 0;
}
