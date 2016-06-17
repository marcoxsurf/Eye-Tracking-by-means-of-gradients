#include <iostream>
#include <opencv2\opencv.hpp>
#include <string>
#include <time.h>

#include "kf_est.h"
#include "utils.h"
#include "eye_op.h"

using namespace cv;
using namespace std;

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

Mat frame;
string main_wnd_name = "Face & Eye Detection";
string right_eye = "Right Eye";
string left_eye = "Left Eye";

void detectFace(Mat frame);
//TODO Sostituire KF con UKF per via del modello nonlineare dell'occhio
KFE leftEye, rightEye;

bool showDetectedLines, recCam;
int frame_width, frame_height;
VideoWriter video;

int main(int argc, char** argv) {
	char *face_file= "haarcascade_frontalface_alt2.xml", *eye_file = "haarcascade_eye.xml";
	int camera=0;
	//init KF for eyes
	leftEye = KFE();
	rightEye = KFE();

	showDetectedLines = true;
	recCam = false;

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
	frame_width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
	frame_height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	namedWindow(main_wnd_name, 1);
	

	double ticks = 0;

	while (true) {
		int c = waitKey(10);
		if ((char)c == 'q') {
			break;
		}
		switch (c) {
		case 'f':
			imwrite("frame.png", frame);
			break;
		case 's':
			showDetectedLines = !showDetectedLines;
			if (showDetectedLines) {
				printf("Show detected lines\n");
			}
			else {
				printf("Hide detected lines\n");
			}
			break;
		case 'r':
			recCam = !recCam;
			if (recCam) {
				printf("Starting rec\n");
				video = VideoWriter("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);
				cout << "Frame Size = " << frame_width << "x" << frame_height << endl;
				if (!video.isOpened()) {
					printf("ERROR: Failed to write the video\n");
					recCam = false;
					break;
				}
				printf("Rec ON\n");
			}
			else {
				video.release();
				printf("Rec OFF\n");
			}
			break;
		}

		// Start time
		elapsed_tick = (double) getTickCount();	//per calcolo fps
		
		//calcolo dt
		double precTick = ticks;
		ticks = (double)cv::getTickCount();
		
		double dT = (ticks - precTick) / freq; //seconds
		cap >> frame;
		if (frame.empty())
			break;
		// Flip the frame horizontally, Windows users might need this
		flip(frame, frame, 1);

		if (leftEye.getFound()) {
			leftEye.setDT(dT);
			circle(frame, leftEye.getCenter(), 2, CV_RGB(255, 0, 0), 1); //?-1
			rectangle(frame, leftEye.getPredRect(), CV_RGB(255, 0, 0), 2);
		}
		if (rightEye.getFound()) {
			rightEye.setDT(dT);
			circle(frame, rightEye.getCenter(), 2, CV_RGB(255, 0, 0), 1); //?-1
			rectangle(frame, rightEye.getPredRect(), CV_RGB(255, 0, 0), 2);
		}

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
		if (recCam) {
			video.write(frame);
		}
	}
	cap.release();
	return 0;
}

void detectFace(Mat frame) {
	// Convert to grayscale and 
	// adjust the image contrast using histogram equalization
	Mat gray;
	vector<Rect> faces, eyes,leye,reye;

	cvtColor(frame, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);

	/*face_cascade.detectMultiScale(gray, faces, 1.1, 2,
		0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT,
		cv::Size(150, 150));*/
	//face_cascade.detectMultiScale(gray, faces, 1.3, 5);
	//face_cascade.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(200, 200));
	
	face_cascade.detectMultiScale(gray, faces, 1.1, 3,
		//CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
		0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT
		//min size
		, cvSize(100, 100)
		//max size
		//, cvSize(300, 300)
	);
	
	//Problema se più facce
	//caso semplice: 1 sola faccia
	// Draw rect on the detected faces
	for (int i = 0; i < faces.size(); i++) {
		Rect face = faces[i];
		//Riduco la sezione della faccia a solo quella di mio interesse
		//IDEA: y = y + h*.2	h = h*.3
		Rect eyeRegion = Rect(face.x, (int)(face.y + face.height*.2), face.width, (int)(face.height*.3));
		if (showDetectedLines) {
			//rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
			rectangle(frame, eyeRegion, Scalar(0, 255, 255), 2);
		}
		Mat faceROI = gray(eyeRegion);
		
		//here kalman's magic
		//eye_cascade
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 5
			, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
			, cvSize(10, 10), cvSize(100, 100));

		//>>>>>>>> Detection
		switch (eyes.size()){
		case 0:
			leftEye.incNotFound();
			rightEye.incNotFound();
			break;
		case 1:
			//sx o dx?
			if (eyes[0].x < faces[i].width / 2) {
				//sx
				//draw left eye
				Rect eye(eyeRegion.x + eyes[0].x, eyeRegion.y + eyes[0].y, eyes[0].width, eyes[0].height);
				if (showDetectedLines) {
					rectangle(frame, eye, Scalar(0, 255, 0), 2);
				}
				leftEye.setMeas(eye);
				rightEye.incNotFound();

			} else {
				//draw right eye
				Rect eye(eyeRegion.x + eyes[0].x, eyeRegion.y + eyes[0].y, eyes[0].width, eyes[0].height);
				if (showDetectedLines) {
					rectangle(frame, eye, Scalar(0, 255, 0), 2);
				}
				rightEye.setMeas(eye);
				leftEye.incNotFound();
				}
			break;
		default:
			//caso 2 = caso >2, se occhi più di due discrimino up,down e sx,dx
			for (int j = 0; j < eyes.size(); j++) {
				//sx o dx?
				/*if (eyes[j].y > (faces[i].height * 0.6 )) {
					continue;
				}*/
				if (eyes[j].x < eyeRegion.width / 2) {
					//sx
					leftEye.resetNotFoundCount();
					//draw left eye
					Rect eye(eyeRegion.x + eyes[j].x, eyeRegion.y + eyes[j].y, eyes[j].width, eyes[j].height);
					if (showDetectedLines) {
						rectangle(frame, eye, Scalar(0, 255, 0), 2);
					}
					leftEye.setMeas(eye);
					Mat leye = gray(eye);
					
					Point leftPupil = findEyeCenter(leye, eye);
				}
				else {
					rightEye.resetNotFoundCount();
					//draw right eye
					Rect eye(eyeRegion.x + eyes[j].x, eyeRegion.y + eyes[j].y, eyes[j].width, eyes[j].height);
					if (showDetectedLines) {
						rectangle(frame, eye, Scalar(0, 255, 0), 2);
					}
					rightEye.setMeas(eye);
					
				}
			}
			break;
		}
		//<<<<<<<< Detection

	}
}

