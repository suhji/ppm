#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "skindetector.h"

/** Constants **/
static int blinkcount=0;
static int centercount = 0;
static int closecount = 0;

/** Function Headers */
void detectAndDisplay(cv::Mat frame);

//namespace란, 관련있는 녀석끼리 모여있는 공간. 

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "C:/cascade/haarcascade_frontalface_alt.xml";
cv::String eye_cascade_name = "C:/cascade_cuda/haarcascade_lefteye_2splits.xml";

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;

std::string main_window_name = "Capture - Face detection";

cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat showImage;

cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
SkinDetector mySkinDetector;
cv::Point leftPupil;
cv::Point rightPupil;
int lx=0, ly=0;
//이전의 eye center값 저장을 위한 변수 
int llx = 0, lly = 0;
/**
* @function main
*/
int main(int argc, const char** argv) {
	cv::Mat frame;
	
	// Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
	//if (!eye_cascade.load(eye_cascade_name)){ printf("--(!)Error loading eye cascade, please change eye_cascade_name in source code.\n"); return -1; };


	cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);


	createCornerKernels();
	ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
		43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);


	// I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
	CvCapture* capture = cvCaptureFromCAM(-1);
	if (capture) {
		while (true) {
			frame = cvQueryFrame(capture);
#else
	cv::VideoCapture capture(1);
	if (capture.isOpened()) {
		while (true) {
			capture.read(frame);
#endif
			// mirror it
			//flipping around y-axis.
			cv::flip(frame, frame, 1);
			frame.copyTo(debugImage);
			frame.copyTo(showImage);
			// Apply the classifier to the frame
			if (!frame.empty()) {
				detectAndDisplay(frame);
			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			imshow(main_window_name, showImage);

			int c = cv::waitKey(10);
			if ((char)c == 'c') {
				printf("Total blink:%d\n", blinkcount);
				break; 
			}
			if ((char)c == 'f') {
				imwrite("frame.png", frame);
			}

			frame.release();
			debugImage.release();
		}
	}

	releaseCornerKernels();

	return 0;
		}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}



	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
	int eye_region_top = face.height * (kEyePercentTop / 100.0);

	cv::Rect eyeregion(face.width*(kEyePercentSide / 100.0),
		eye_region_top, eye_region_width, eye_region_height);
	
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0),
		eye_region_top, eye_region_width, eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0),
		eye_region_top, eye_region_width, eye_region_height);

	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");

	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	rectangle(debugFace, leftRightCornerRegion, 200);
	rectangle(debugFace, leftLeftCornerRegion, 200);
	rectangle(debugFace, rightLeftCornerRegion, 200);
	rectangle(debugFace, rightRightCornerRegion, 200);
	
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;


	//이전 frame의 eyecenter 값을 llx, lly에 저장. 
	if (centercount == 0){
		llx = leftPupil.x;
		lly = leftPupil.y;

		lx = llx;
		ly = lly;
		centercount++;
	}
	else{
		llx = lx;
		lly = ly;

		lx = leftPupil.x;
		ly = leftPupil.y;
	}
	

	faceROI.release();
	debugFace.release();
}


cv::Mat findSkin(cv::Mat &frame) {
	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows, frame.cols, CV_8U);

	//frame:BGR, input:YCrCb
	cvtColor(frame, input, CV_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		//    uchar *Or = output.ptr<uchar>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) {
			cv::Vec3b ycrcb = Mr[x];
			//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
			if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
				Or[x] = cv::Vec3b(0, 0, 0);
			}
		}
	}

	const cv::Vec3b *tr = input.ptr<cv::Vec3b>(lly);
		cv::Vec3b *er = frame.ptr<cv::Vec3b>(lly);
		cv::Vec3b tycrcb = tr[llx];
		if (skinCrCbHist.at<uchar>(tycrcb[1], tycrcb[2]) == 0) {
			printf("open\n");
			printf("blink%d\n", blinkcount);
		}
		else
		{
			blinkcount++;
			printf("closed\n");
			printf("blink%d\n",blinkcount);

		}

	input.release();
	return output;	
}

/**
* @function detectAndDisplay
*/
void detectAndDisplay(cv::Mat frame) {
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eye;
	//cv::Mat frame_gray;

	std::vector<cv::Mat> rgbChannels(3);	//vector array rgbChannels선언
	cv::split(frame, rgbChannels);			//frame을 rgbChannels로 reallocate?
	cv::Mat frame_gray = rgbChannels[2];    //n dimensional array class Mat. 0->1->2로 갈수록 밝아 지는 듯.  grayscale을 뽑아내는 듯.


	//-- Detect faces. 감지된 물체 list of rectangles로 반환 
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));
	
	//-- Show what you got
	if (faces.size() > 0) {
		findEyes(frame_gray, faces[0]);
		findSkin(debugImage(faces[0]));
	
	}
	
	frame_gray.release();
}
