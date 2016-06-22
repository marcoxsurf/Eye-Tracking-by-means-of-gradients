#pragma once
#include <opencv2\opencv.hpp>
using namespace cv;

Point findEyeCenter(Mat face, Rect eye);

void findCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
Point unscalePoint(Point p, Rect origSize);
bool isInsideCircle(int x, int y, int cx, int cy, int r);
Mat floodKillEdges(Mat &mat);