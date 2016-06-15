#include "kf_eye.h"

using namespace std;
using namespace cv;

void KF::setVars() {
	found = false;
	notFoundCount = 0;
	kf = KalmanFilter(stateSize, measSize, contrSize, type);
	// [x,y,v_x,v_y,w,h]
	Mat state(stateSize, 1, type);
	// [z_x,z_y,z_w,z_h]
	Mat meas(measSize, 1, type);
	initSMNMatrix();
}

//default costruttore
KF::KF() {
	stateSize = 6;
	measSize = 4;
	contrSize = 0;
	type = CV_32F;
	setVars();
}

KF::KF(int _stateSize, int _measSize, int _contrSize, unsigned int _type) {
	stateSize=_stateSize;
	measSize = _measSize;
	contrSize = _contrSize;
	type = _type;
	setVars();
}

/*
	Init State Meas and Noise Matrix (Q,R)
*/
void KF::initSMNMatrix() {
	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0  0 0 ]
	// [ 0 1 0  dT 0 0 ]
	// [ 0 0 1  0  0 0 ]
	// [ 0 0 0  1  0 0 ]
	// [ 0 0 0  0  1 0 ]
	// [ 0 0 0  0  0 1 ]
	setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2f;
	kf.processNoiseCov.at<float>(7) = 1e-2f;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2f;
	kf.processNoiseCov.at<float>(35) = 1e-2f;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
}

void KF::incNotFound() {
	notFoundCount++;
	printf("notFoundCount: %d", notFoundCount);
	if (notFoundCount >= 100) {
		found = false;
	}
}

void KF::resetNotFoundCount() {
	notFoundCount = 0;
}

bool KF::getFound() {
	return found;
}
void KF::setFound(bool _found) {
	found = _found;
}

void KF::setDT(double dt) {
	// >>>> Matrix A
	kf.transitionMatrix.at<float>(2) = (float)dt;
	kf.transitionMatrix.at<float>(9) = (float)dt;
	// <<<< Matrix A
	printf("dT: %f", dt);
	//ad ogni step, predict
	predict();
}

void KF::predict() {
	state = kf.predict();
	cout << "State post:" << endl << state << endl;
}

Mat KF::getState() {
	return state;
}

Rect KF::getPredRect() {
	Rect predRect;
	predRect.width = (int) state.at<float>(4);
	predRect.height = (int)state.at<float>(5);
	predRect.x = (int) state.at<float>(0) - predRect.width / 2;
	predRect.y = (int) state.at<float>(1) - predRect.height / 2;
	return predRect;
}

Point KF::getCenter() {
	Point center;
	center.x = (int)state.at<float>(0);
	center.y = (int)state.at<float>(1);
	return center;
}

void KF::setMeas(Rect rect) {
	if (rect.height == 0)
		return;
	notFoundCount = 0;
	//TODO Possible correzione da effettuare qui
	meas.at<float>(0) = (float)rect.x + rect.width / 2;
	meas.at<float>(1) = (float)rect.y + rect.height / 2;
	meas.at<float>(2) = (float)rect.width;
	meas.at<float>(3) = (float)rect.height;

	if (!found) {
		// First detection!
		kf.errorCovPre.at<float>(0) = 1; // px
		kf.errorCovPre.at<float>(7) = 1; // px
		kf.errorCovPre.at<float>(14) = 1;
		kf.errorCovPre.at<float>(21) = 1;
		kf.errorCovPre.at<float>(28) = 1; // px
		kf.errorCovPre.at<float>(35) = 1; // px

		state.at<float>(0) = meas.at<float>(0);
		state.at<float>(1) = meas.at<float>(1);
		state.at<float>(2) = 0;
		state.at<float>(3) = 0;
		state.at<float>(4) = meas.at<float>(2);
		state.at<float>(5) = meas.at<float>(3);
		
		found = true;
	}
	else
		kf.correct(meas); // Kalman Correction

	cout << "Measure matrix:" << endl << meas << endl;
}

KF::~KF() {
	
}

