#include <iostream>
#include <opencv2\opencv.hpp>
#include <string>
#include <time.h>


#include "utils.h"

using namespace cv;
using namespace std;

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;

cv::Mat frame;
std::string main_wnd_name = "Face & Eye Detection";

void detectFace(cv::Mat frame);


int main(int argc, char** argv) {
	char *face_file= "haarcascade_frontalface_alt2.xml", *eye_file = "haarcascade_eye.xml";
	int camera=0;
	
	double elapsed_time, elapsed_tick;
	double freq=getTickFrequency();

	face_cascade.load(face_file);
	eye_cascade.load(eye_file);
	
	// Open webcam
	VideoCapture cap(camera);
	// Check if everything is ok
	if (face_cascade.empty()) {
		printf("Errore con il file: %s", face_file);
		return 1;
	}
	if (eye_cascade.empty()) {
		printf("Errore con il file: %s", eye_file);
		return 1;
	}
	if (!cap.isOpened()) {
		printf("Errore con la webcam: %d", camera);
		return 1;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	Mat eye_tpl;
	Rect eye_bb;
	namedWindow(main_wnd_name, 1);
	while (waitKey(15) != 'q') {
		// Start time
		//time(&start);
		elapsed_tick = (double) getTickCount();

		cap >> frame;
		if (frame.empty())
			continue;
		// Flip the frame horizontally, Windows users might need this
		flip(frame, frame, 1);

		//Copy frame to image to display
		frame.copyTo(frame);

		detectFace(frame);
		
		//Calcolo tempo per un frame
		// End Time
		elapsed_tick = getTickCount() - elapsed_tick;
		elapsed_time = elapsed_tick / freq;
		// 1 frm in a msec, quanti frame al secondo?
		// 1f : a = x : 1
		// x = 1f*1/a		
		frame = addFPStoFrame(frame, elapsed_time);
		imshow(main_wnd_name, frame);
	}
	cap.release();
	return 0;
}

void detectFace(cv::Mat frame) {
	// Convert to grayscale and 
	// adjust the image contrast using histogram equalization
	Mat gray;
	vector<Rect> faces, eyes;

	cvtColor(frame, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);

	/*face_cascade.detectMultiScale(gray, faces, 1.1, 2,
		0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT,
		cv::Size(150, 150));*/
	//face_cascade.detectMultiScale(gray, faces, 1.3, 5);
	//face_cascade.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(200, 200));
	
	face_cascade.detectMultiScale(gray, faces, 1.1, 10,
	CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
	, cvSize(100, 100), cvSize(300, 300));
	

	// Draw rect on the detected faces
	for (int i = 0; i < faces.size(); i++) {
		rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
		//potrei pensare di ridurre faces[i]
		//invece di dare tutto la faccia, solo la fascia degli occhi
		//TODO Reduce face region for eye catching
		//params da paper Roberto Valenti
		//eye centers are always contained within 2 regions starting from
		//20%x30% (left eye), and 60%x30% (right eye) of the face region
		//with dimensions of 25%x20% of the latter
		//Give only half face to cascade
		Rect halfFace(faces[i].x, faces[i].y, faces[i].width, faces[i].height/2);
		Mat halfFaceROI = gray(halfFace);
		//here kalman's magic
		//eye_cascade
		eye_cascade.detectMultiScale(halfFaceROI, eyes, 1.1, 8
			, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
			, cvSize(10, 10), cvSize(100, 100));
		for (int j = 0; j < eyes.size(); j++) {
			//effettuo traslazione dei punti per disagnare su frame
			Rect eye(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);
			rectangle(frame, eye, Scalar(255, 0, 0), 1);
		}
	}
}

