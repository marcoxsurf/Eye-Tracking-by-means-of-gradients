#include <opencv2\opencv.hpp>
#include <sstream>

#include "utils.h"

using namespace cv;
using namespace std;

Mat	addFPStoFrame(Mat frame, double elapsed_time) {
	ostringstream strs;
	double nframe = floor((1 / elapsed_time)*100.0) / 100.0;
	strs << nframe;
	string fps = "FPS: " + strs.str();
	Size textsize = getTextSize(fps, FONT_HERSHEY_COMPLEX, 1, 3, 0);
	Point org((640 - textsize.width), (480 - textsize.height));
	int lineType = 8;
	putText(frame, fps, org, FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 3, lineType);
	return frame;
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
	Scalar stdMagnGrad, meanMagnGrad;
	meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
	Mat mags(matX.rows, matX.cols, CV_64F);
	for (int y = 0; y < matX.rows; ++y) {
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}