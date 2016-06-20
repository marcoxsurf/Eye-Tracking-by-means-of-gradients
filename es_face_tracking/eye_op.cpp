#include <opencv2\opencv.hpp>

#include "eye_op.h"
#include "constants.h"
#include "utils.h"

using namespace cv;

Mat computeMatXGradient(const cv::Mat &mat) {
	cv::Mat out(mat.rows, mat.cols, CV_64F);
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

Point findEyeCenter(Mat face, Rect eye) {
	Mat eyeROIUnscaled = face(eye);
	Mat eyeROI;
	Mat gradientX, gradientY, ggX;
	Mat mags;
	//scaleToFastSize(eyeROIUnscaled, eyeROI);
	resize(eyeROIUnscaled, eyeROI, Size(kFastEyeWidth, kFastEyeWidth));
	// draw eye region
	//rectangle(face, eye, 1234);
	//-- Find the gradient
	//Sobel(eyeROI, gradientX, CV_64F, 1, 0, 3);
	//Sobel(eyeROI, gradientY, CV_64F, 0, 1, 3);
	gradientX = computeMatXGradient(eyeROI);
	gradientY = computeMatXGradient(eyeROI.t()).t();
	//Scharr( eyeROI, gradientX, CV_64F, 1, 0, 1, 0, BORDER_DEFAULT );
	//Scharr( eyeROI, gradientY, CV_64F, 1, 0, 1, 0, BORDER_DEFAULT );
	//Se Sobel non dovesse funzionare cambiare con questa
	
	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	//magnitude(gradientX, gradientY, mags);
	mags = matrixMagnitude(gradientX, gradientY);
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	//normalize
	//normalize(eyeROI, gradientX, gradientThresh);
	//normalize(eyeROI, gradientY, gradientThresh);
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


	//imshow("gradientx", gradientX);
	//-- Create a blurred and inverted image for weighting
	Mat weight;
	GaussianBlur(eyeROI, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
	weight = 255 - weight;
	imshow("Eye", weight);
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
	double numGradients = (weight.rows*weight.cols);
	Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / numGradients);
	//-- Find the maximum point
	Point maxP;
	double maxVal;
	minMaxLoc(out, NULL, &maxVal, NULL, &maxP);
	return unscalePoint(maxP, eye);
}

void findCentersFormula(int x, int y, const Mat &weight, double gx, double gy, Mat &out) {
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
			Or[cx] += dotProduct * dotProduct * (Wr[cx] / kWeightDivisor);
		}
	}
}

Point unscalePoint(Point p, Rect origSize) {
	float ratio = (((float)kFastEyeWidth) / origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return Point(x, y);
}