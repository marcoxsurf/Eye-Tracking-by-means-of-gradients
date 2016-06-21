#include "kf_est.h"

using namespace std;
using namespace cv;

void KFE::setVars() {
	found = false;
	notFoundCount = 0;
	kf = KalmanFilter(stateSize, measSize, contrSize, type);
	// [x,y,v_x,v_y,a_x,a_y,w,h]
	state = Mat(stateSize, 1, type);
	// [z_x,z_y,z_w,z_h]
	meas = Mat(measSize, 1, type);
	initSMNMatrix();
}

//default costruttore
KFE::KFE() {
	Ak = 0.08f;
	wk = 3.1415926535f;
	stateSize = 8;
	measSize = 4;
	contrSize = 0;
	type = CV_32F;
	f_width = 640;
	f_height = 480;
	setVars();
}

KFE::KFE(int _stateSize, int _measSize, int _contrSize, unsigned int _type) {
	stateSize = _stateSize;
	measSize = _measSize;
	contrSize = _contrSize;
	type = _type;
	setVars();
}

void KFE::setWH(int width, int height) {
	f_width = width;
	f_height = height;
}

void KFE::setDT(double dt) {
	// >>>> Matrix A
	//x
	//v_x
	kf.transitionMatrix.at<float>(2) = (float)dt;
	//a_x
	kf.transitionMatrix.at<float>(4) = (float)(.5*dt*dt);
	//y
	//v_y
	kf.transitionMatrix.at<float>(11) = (float)dt;
	//a_y
	kf.transitionMatrix.at<float>(13) = (float)(.5*dt*dt);
	//v_x_y
	kf.transitionMatrix.at<float>(20) = (float)(Ak*sin(wk*dt));
	kf.transitionMatrix.at<float>(29) = (float)(Ak*sin(wk*dt));
	//a_x_y
	kf.transitionMatrix.at<float>(36) = (float)(Ak*cos(wk*dt));
	kf.transitionMatrix.at<float>(45) = (float)(Ak*cos(wk*dt));
	// <<<< Matrix A
	//printf("dT: %f", dt);
	//ad ogni step, predict
	predict();
}

/*
Init State Meas and Noise Matrix (Q,R)
*/
void KFE::initSMNMatrix() {
	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0 1/2*dt^2 0 0 0 ]		--> xk
	// [ 0 1 0 dT 0 1/2*dt^2 0 0 ]		--> yk
	// [ 0 0 1 0 Ak*sin(wkdt) 0 0 0 ]	--> v_x
	// [ 0 0 0 1 0 Ak*sin(wkdt) 0 0 ]	--> v_y
	// [ 0 0 0 0 Ak*wk*cos(wkdt) 0 0 0 ]--> a_x
	// [ 0 0 0 0 0 Ak*wk*cos(wkdt) 0 0 ]--> a_y
	// [ 0 0 0 0 0 0 1 0]				--> w
	// [ 0 0 0 0 0 0 0 1]				--> h
	setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 0 0 ]
	// [ 0 0 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(9) = 1.0f;
	kf.measurementMatrix.at<float>(22) = 1.0f;
	kf.measurementMatrix.at<float>(31) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0    0    0  ]
	// [ 0    Ey  0     0     0    0    0    0  ]
	// [ 0    0   Ev_x  0     0    0    0    0  ]
	// [ 0    0   0     Ev_y  0    0    0    0  ]
	// [ 0    0   0     0     Ea_x 0    0    0  ]
	// [ 0    0   0     0     0    Ea_y 0    0  ]
	// [ 0    0   0     0     0    0    Ew   0  ]
	// [ 0    0   0     0     0    0    0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2f;
	kf.processNoiseCov.at<float>(9) = 1e-2f;
	kf.processNoiseCov.at<float>(18) = 2.0f; //5.0f
	kf.processNoiseCov.at<float>(27) = 2.0f; //5.0f
	kf.processNoiseCov.at<float>(36) = 2.0f; //5.0f
	kf.processNoiseCov.at<float>(45) = 2.0f; //5.0f
	kf.processNoiseCov.at<float>(54) = 1e-2f;
	kf.processNoiseCov.at<float>(63) = 1e-2f;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
}

/*
Inserisco nel modello l'occhio trovato da cascade detection
*/
void KFE::setMeas(Rect rect) {
	if (rect.height == 0)
		return;
	notFoundCount = 0;
	meas.at<float>(0) = (float)rect.x + rect.width / 2;
	meas.at<float>(1) = (float)rect.y + rect.height / 2;
	meas.at<float>(2) = (float)rect.width;
	meas.at<float>(3) = (float)rect.height;

	if (!found) {
		// First detection!
		kf.errorCovPre.at<float>(0) = 1; // px
		kf.errorCovPre.at<float>(9) = 1; // py
		kf.errorCovPre.at<float>(18) = 1;
		kf.errorCovPre.at<float>(27) = 1;
		kf.errorCovPre.at<float>(36) = 1; 
		kf.errorCovPre.at<float>(45) = 1; 
		kf.errorCovPre.at<float>(54) = 1; // pw
		kf.errorCovPre.at<float>(63) = 1; // ph

		state.at<float>(0) = meas.at<float>(0);
		state.at<float>(1) = meas.at<float>(1);
		state.at<float>(2) = 0;
		state.at<float>(3) = 0;
		state.at<float>(4) = 0;
		state.at<float>(5) = 0;
		state.at<float>(6) = meas.at<float>(2);
		state.at<float>(7) = meas.at<float>(3);

		found = true;
	}
	else
		// Kalman Correction
		kf.correct(meas); 

	//cout << "Measure matrix:" << endl << meas << endl;
}

void KFE::incNotFound() {
	notFoundCount++;
	//printf("notFoundCount: %d", notFoundCount);
	if (notFoundCount >= 100) {
		found = false;
	}
}

void KFE::resetNotFoundCount() {
	notFoundCount = 0;
}

bool KFE::getFound() {
	return found;
}
void KFE::setFound(bool _found) {
	found = _found;
}

void KFE::predict() {
	state = kf.predict();
	//cout << "State post:" << endl << state << endl;
}

Mat KFE::getState() {
	return state;
}

Rect KFE::getPredRect() {
	Rect predRect;
	predRect.width = (int)state.at<float>(6);
	predRect.height = (int)state.at<float>(7);
	//se rettangolo fuori frame ritorno rettangolo centrato
	if (
		// x>=0 && y>=0
		((int)state.at<float>(0) >= 0) && ((int)state.at<float>(1) >= 0) &&
		// x<=f_height - height 
		((int)state.at<float>(0) <= f_height - (int)state.at<float>(7)) &&
		//y<=f_width - width
		((int)state.at<float>(1) <= f_width - (int)state.at<float>(6))
		) {
		predRect.x = (int)state.at<float>(0) - predRect.width / 2;
		predRect.y = (int)state.at<float>(1) - predRect.height / 2;
	}
	else {
		predRect.x = f_height / 2;
		predRect.y = f_width / 2;
	}
	return predRect;
}

Point KFE::getCenter() {
	Point center;
	center.x = (int)state.at<float>(0);
	center.y = (int)state.at<float>(1);
	return center;
}

KFE::~KFE() {}

