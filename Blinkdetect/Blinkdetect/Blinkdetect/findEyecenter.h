#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include "opencv2\imgproc\imgproc.hpp"

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);

bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p, int rows, int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

#endif