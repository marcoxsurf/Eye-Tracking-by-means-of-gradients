#pragma once
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class KFE {
	double Ak;
	double wk;

	int stateSize;
	int measSize;
	int contrSize;
	//CV_32F is float - the pixel can have any value between 0-1.0
	unsigned int type;

	KalmanFilter kf;

	Mat state, meas;

	bool found;
	int notFoundCount;

	int f_width, f_height;

public:
	KFE();
	KFE(int _stateSize, int _measSize, int _contrSize, unsigned int _type);
	~KFE();

	void setDT(double dt);
	Mat getState();
	Rect getPredRect();
	Point getCenter();

	void setMeas(Rect rect);

	void incNotFound();
	void resetNotFoundCount();

	bool getFound();
	void setFound(bool _found);

	void setWH(int width, int height);
private:
	void initSMNMatrix();
	void predict();
	void setVars();
};