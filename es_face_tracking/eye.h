#pragma once
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class Eye {
	Point pupil;

public:
	Eye();

	void setPupil(Point _pupil);
	Point getPupil();
};