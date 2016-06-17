#pragma once
#include <opencv2\opencv.hpp>
using namespace cv;

Point findEyeCenter(Mat face, Rect eye);

void scaleToFastSize(const cv::Mat &src, cv::Mat &dst);

Mat computeMatXGradient(const Mat &mat);
//void normalize(Mat eyeROI, double gradientThresh, Mat mags, Mat &gradientX, Mat &gradientY);