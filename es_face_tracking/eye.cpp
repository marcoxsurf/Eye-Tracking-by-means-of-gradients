#include <opencv2\opencv.hpp>
#include <iostream>
#include "eye.h"

using namespace cv;
using namespace std;

Eye::Eye() {

}

Point Eye::getPupil() {
	return pupil;
}

void Eye::setPupil(Point _pupil) {
	pupil = _pupil;
	//creo rettangoli dx e sx della pupilla

}