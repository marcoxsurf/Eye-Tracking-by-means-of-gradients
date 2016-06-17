#pragma once
#include <opencv2\opencv.hpp>

/*
Funzione per aggiungere la stringa FPS: al frame in output
@msec misura il tempo in sec tra un frame e l'altro
*/
cv::Mat	addFPStoFrame(cv::Mat frame, double durata);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
