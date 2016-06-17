#include <opencv2\opencv.hpp>

#include "eye_op.h"
#include "constants.h"
#include "utils.h"

using namespace cv;

Point findEyeCenter(Mat face, Rect eye) {
	Mat eyeROIUnscaled = face(eye);
	Mat eyeROI;
	Mat gradientX, gradientY;
	Mat mags;
	//scaleToFastSize(eyeROIUnscaled, eyeROI);
	resize(eyeROIUnscaled, eyeROI, Size(kFastEyeWidth, kFastEyeWidth));
	// draw eye region
	rectangle(face, eye, 1234);
	//-- Find the gradient
	Sobel(eyeROI, gradientX, CV_64F, 1, 0, 3);
	Sobel(eyeROI, gradientY, CV_64F, 1, 0, 3);
	//Se Sobel non dovesse funzionare cambiare con questa
	/*cv::Mat gradientX = computeMatXGradient(eyeROI);
	cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();*/
	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	magnitude(gradientX, gradientY, mags);
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	//normalize
	normalize(eyeROI, gradientX, gradientThresh);
	normalize(eyeROI, gradientY, gradientThresh);
	
	//-- Create a blurred and inverted image for weighting
	Mat weight;
	GaussianBlur(eyeROI, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
	weight = 255 - weight;
	imshow("Eye", weight);

	return Point(0, 0);
}

/*
void normalize(Mat eyeROI, double gradientThresh, Mat mags, Mat &gradientX, Mat &gradientY) {
	for (int y = 0; y < eyeROI.rows; ++y) {
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh) {
				Xr[x] = gX / magnitude;
				Yr[x] = gY / magnitude;
			}
			else {
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}
}*/

Mat computeMatXGradient(const Mat &mat) {
	Mat out(mat.rows, mat.cols, CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {
			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}

	return out;
}

void scaleToFastSize(const Mat &src, Mat &dst) {
	resize(src, dst, Size(kFastEyeWidth, (int)(((float)kFastEyeWidth) / src.cols) * src.rows));
}