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
	//imshow("Eye", weight);
	//Inizia l'algoritmo
	Mat outSum = Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);
	// for each possible gradient location
	// Note: these loops are reversed from the way the paper does them
	// it evaluates every possible center for each gradient location instead of
	// every possible gradient location for every center.
	//printf("Eye Size: %ix%i\n", outSum.cols, outSum.rows);
	for (int y = 0; y < weight.rows; ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			findCentersFormula(x, y, weight, gX, gY, outSum);
		}
	}
	// scale all the values down, basically averaging them
	/*double numGradients = (weight.rows*weight.cols);
	Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / numGradients);
	imshow("Eye", out);*/
	return Point(0, 0);
}

void findCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out) {
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		double *Or = out.ptr<double>(cy);
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0, dotProduct);
			// square and multiply by the weight
			Or[cx] += dotProduct * dotProduct * Wr[cx];
		}
	}
}