#pragma once

//Parameters
const int kFastEyeWidth = 50;
const double kGradientThreshold = 50.0;
const int kWeightBlurSize = 5;
const float kWeightDivisor = 1.0;

const int radiusEye = 10;	//10 pixel

// Postprocessing
const bool kEnablePostProcess = true;
const double kPostProcessThreshold = .97;