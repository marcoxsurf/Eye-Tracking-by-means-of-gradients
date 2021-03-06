#include <iostream>
#include <opencv2\opencv.hpp>
#include <string>
#include <time.h>

#include "kf_est.h"
#include "utils.h"
#include "eye_center.h"
#include "eye_corner.h"
#include "constants.h"

using namespace cv;
using namespace std;

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

Mat frame;
string main_wnd_name = "Face & Eye Detection";
string right_eye = "Right Eye";
string left_eye = "Left Eye";

void detectFace(Mat frame);
void printHelp();
//TODO Sostituire KF con UKF per via del modello nonlineare dell'occhio
KFE leftEye, rightEye;
Point leftPupil, rightPupil;

bool showDetectedLines, showKalmanLines, recCam, findEyeC;
int frame_width, frame_height;
VideoWriter video;

int main(int argc, char** argv) {
	char *face_file= "haarcascade_frontalface_alt2.xml", *eye_file = "haarcascade_eye.xml";
	int camera=0;
	//init KF for eyes
	leftEye = KFE();
	rightEye = KFE();

	showDetectedLines = true;
	showKalmanLines = true;

	recCam = false;
	findEyeC = false;

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

	rightEye.setWH(frame_width, frame_height);
	leftEye.setWH(frame_width, frame_height);

	namedWindow(main_wnd_name, 1);
	printHelp();

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
		case 'd':
			showDetectedLines = !showDetectedLines;
			if (showDetectedLines) {
				printf("Show detected lines\n");
			}
			else {
				printf("Hide detected lines\n");
			}
			break;
		case 'k':
			showKalmanLines = !showKalmanLines;
			if (showKalmanLines) {
				printf("Show Kalman lines\n");
			}
			else {
				printf("Hide Kalman lines\n");
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
		case 'e':
			findEyeC = !findEyeC;
			printf("Find eye: %d\n",findEyeC);
			break;
		case 'h':
			printHelp();
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
		GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		equalizeHist(gray, gray);
		//imshow("gray", gray);
		//TODO controllo se box predetto � fuori frame 
		if (leftEye.getFound()) {
			leftEye.setDT(dT);
			Rect le = leftEye.getPredRect();
			if (showKalmanLines) {
				circle(frame, leftEye.getCenter(), 2, CV_RGB(255, 0, 0), -1);
				rectangle(frame, le, CV_RGB(255, 0, 0), 2);
			}
			if (findEyeC) {
				if (le.width > 0 && le.height > 0) {
					leftPupil = findEyeCenter(gray, le);
					leftPupil.x += le.x;
					leftPupil.y += le.y;
					circle(frame, leftPupil, 3, CV_RGB(0, 0, 255));
				}
			}
		}
		if (rightEye.getFound()) {
			rightEye.setDT(dT);
			Rect re = rightEye.getPredRect();
			if (showKalmanLines) {
				circle(frame, rightEye.getCenter(), 2, CV_RGB(255, 0, 0), -1);
				rectangle(frame, re, CV_RGB(255, 0, 0), 2);
			}
			if (findEyeC) {
				if (re.width > 0 && re.height > 0) {
					rightPupil = findEyeCenter(gray, re);
					rightPupil.x += re.x;
					rightPupil.y += re.y;
					circle(frame, rightPupil, 2, CV_RGB(0, 0, 255));
				}
			}
		}
		if (findEyeC && leftEye.getFound() && rightEye.getFound()) {
			//devo creare le 4 aree intorno al eye corner per essere rilevate
			// da un filtro Gabor
			int a = rightPupil.x - leftPupil.x; //dist tra occhi
			//secondo http://stackoverflow.com/questions/9645871/how-to-perform-stable-eye-corner-detection
			//b = larghezza dell'occhio, � proporzionale ad a
			//c = altezza occhio, � proporzionale ad a
			//mie misure, a = 4cm, b=1.73->2cm, c=0.8->1cm
			double k_b = 1.8;
			double k_c = 4.0;
			int b = (int)(a / k_b);
			int c = (int)(a / k_c);

			Rect r1, r2, r3, r4;
			r1 = Rect(leftPupil.x - b / 2, leftPupil.y - c/2, b/2, c);
			r2 = Rect(leftPupil.x, leftPupil.y -c/2 , b/2, c);
			r3 = Rect(rightPupil.x - b / 2, rightPupil.y - c/2, b/2, c);
			r4 = Rect(rightPupil.x, rightPupil.y -c/2, b/2, c);
			rectangle(frame, r1, CV_RGB(180, 180, 0), 1);
			rectangle(frame, r2, CV_RGB(180, 180, 0), 1);
			rectangle(frame, r3, CV_RGB(180, 180, 0), 1);
			rectangle(frame, r4, CV_RGB(180, 180, 0), 1);
			//ora ho le regioni per andare a trovare i corner dell'occhio
		}
		detectFace(gray);
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

void detectFace(Mat gray) {
	// Convert to grayscale and 
	// adjust the image contrast using histogram equalization
	vector<Rect> faces, eyes,leye,reye;
	
	/*face_cascade.detectMultiScale(gray, faces, 1.1, 2,
		0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT,
		cv::Size(150, 150));*/
	//face_cascade.detectMultiScale(gray, faces, 1.3, 5);
	//face_cascade.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(200, 200));
	
	face_cascade.detectMultiScale(gray, faces, 1.1, 3,
		//CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
		0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT
		//min size
		, cvSize(150, 150)
		//max size
		//, cvSize(300, 300)
	);
	
	//Problema se pi� facce
	//caso semplice: 1 sola faccia
	// Draw rect on the detected faces
	for (int i = 0; i < faces.size(); i++) {
		Rect face = faces[i];
		//Riduco la sezione della faccia agli occhi
		//IDEA: y = y + h*.2	h = h*.3
		Rect eyeRegion = Rect(face.x, (int)(face.y + face.height*.2), face.width, (int)(face.height*.3));
		if (showDetectedLines) {
			rectangle(frame, eyeRegion, Scalar(0, 255, 255), 2);
		}
		Mat faceROI = gray(eyeRegion);
		
		//eye_cascade
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 5
			, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING
			, cvSize(10, 10)
			, cvSize(80, 80)
		);
		//cascade pu� fornire pi� finestre per lo stesso occchio
		//TODO somma di aeree continue
		//Ora seleziono solo la prima area trovata per occhio
		//here kalman's magic
		//>>>>>>>> Detection
		switch (eyes.size()){
		case 0:
			leftEye.incNotFound();
			rightEye.incNotFound();
			break;
		case 1:
			//sx o dx?
			if (eyes[0].x < eyeRegion.width / 2) {
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
			//caso 2 = caso >2, se occhi pi� di due discrimino up,down e sx,dx
			bool foundR = false, foundL = false;
			for (int j = 0; j < eyes.size(); j++) {
				//sx o dx?
				if (!foundL && eyes[j].x < eyeRegion.width / 2) {
					//sx
					foundL = true;
					leftEye.resetNotFoundCount();
					//draw left eye
					Rect eye(eyeRegion.x + eyes[j].x, eyeRegion.y + eyes[j].y, eyes[j].width, eyes[j].height);
					if (showDetectedLines) {
						rectangle(frame, eye, Scalar(0, 255, 0), 2);
					}
					leftEye.setMeas(eye);
				}
				if (!foundR && eyes[j].x > eyeRegion.width / 2) {
					//dx
					foundR = true;
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

void printHelp() {
	printf("Tasti implementati: \n");
	printf("q - Exit\n\n");
	printf("d - Show/Hide detected lines\n");
	printf("e - Attiva calcolo centro occhio\n");
	printf("f - Print frame\n");
	printf("k - Show/Hide kalman lines\n");
	printf("r - Start/Stop record main frame\n");
}