#include <iostream>
#include <opencv2\opencv.hpp>
#include <queue>

#include "eye_center.h"
#include "constants.h"
#include "utils.h"

using namespace cv;
using namespace std;

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
	//Eye detection ritorna un box centrato sull'occhio
	//Posso sfruttare questa info per ridurre l'area in cui cercare
	//-- Find the gradient
	gradientX = computeMatXGradient(eyeROI);
	gradientY = computeMatXGradient(eyeROI.t()).t();
	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	magnitude(gradientX, gradientY, mags);
	//mags = matrixMagnitude(gradientX, gradientY);
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	//normalize
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
	//imshow("Eye", weight);
	//Inizia l'algoritmo
	
	Mat outSum = Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);
	// for each possible gradient location
	// Note: these loops are reversed from the way the paper does them
	// it evaluates every possible center for each gradient location instead of
	// every possible gradient location for every center.
	//for (int y = 0; y < weight.rows; ++y) {
	for (int y = (int)(kFastEyeWidth / 2 - radiusEye); y < (int)(kFastEyeWidth / 2 + radiusEye); ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = (int)(kFastEyeWidth / 2 - radiusEye); x < (int)(kFastEyeWidth / 2 + radiusEye); ++x) {
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			//se x,y appartengono alla maschera eseguo 
			if (isInsideCircle(x, y, kFastEyeWidth/2, kFastEyeWidth / 2, radiusEye)) {
				findCentersFormula(x, y, weight, gX, gY, outSum);
			}
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
	//cout<<format(out, Formatter::FMT_MATLAB)<<endl;
	//-- Flood fill the edges
	Mat floodClone;
	// remove all remaining values that are connected to one of the borders
	double floodThresh = maxVal * kPostProcessThreshold;
	//threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
	threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
	Mat mask = floodKillEdges(floodClone);
	//imshow(debugWindow,out);
	minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);
	//cout << format(floodClone, Formatter::FMT_MATLAB) << endl;
	return unscalePoint(maxP, eye);
}

bool inMat(Point p, int rows, int cols) {
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
	return inMat(np, mat.rows, mat.cols);
}

Mat floodKillEdges(Mat &mat) {
	rectangle(mat, Rect(0, 0, mat.cols, mat.rows), 255);
	Mat mask(mat.rows, mat.cols, CV_8U, 255);
	queue<Point> toDo;
	toDo.push(Point(0, 0));
	while (!toDo.empty()) {
		Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f) {
			continue;
		}
		// add in every direction
		Point np(p.x + 1, p.y); // right
		if (floodShouldPushPoint(np, mat)) 
			toDo.push(np);
		np.x = p.x - 1; np.y = p.y; // left
		if (floodShouldPushPoint(np, mat)) 
			toDo.push(np);
		np.x = p.x; np.y = p.y + 1; // down
		if (floodShouldPushPoint(np, mat)) 
			toDo.push(np);
		np.x = p.x; np.y = p.y - 1; // up
		if (floodShouldPushPoint(np, mat)) 
			toDo.push(np);
		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
	return mask;
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
			if (!isInsideCircle(cx, cy, kFastEyeWidth / 2, kFastEyeWidth / 2, radiusEye)) {
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
	int x = (int)round(p.x / ratio);
	int y = (int)round(p.y / ratio);
	return Point(x, y);
}

bool isInsideCircle(int x, int y, int cx, int cy, int r) {
	return ((x - cx)*(x - cx) + (y - cy)*(y - cy)) <= r*r;
}