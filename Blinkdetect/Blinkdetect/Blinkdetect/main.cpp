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
#include <time.h>
#include<Windows.h>

#include "constants.h"
#include "findEyeCenter.h"

using namespace cv;

static int blinkcount = 0;
static int preblinkcount = 0;
static int centercount = 0;
static int closecount = 0;
static int blinkinput = 100; //���� ���ݿ� blinkinput�� �Ѿ�� ���˸��� ��
static int flag = 0; //open-close ���� ��ȭ üũ�� ���� ���� flag 
static int skinpixel = 0; //�� �κ� skin pixel���� count �ϱ� ���� ���� 
static int preskinpixel = 0; //�� �κ� skin pixel���� count �ϱ� ���� ����
static int framecount = 0;
static int motioncount = 0;
int lx = 0, ly = 0;
//������ eye center�� ������ ���� ���� 
int llx = 0, lly = 0;

//timecounting�� ���� ����
int pre, current = 0;
int timeflag = 0;

int blinkflag = 0;

void detectAndDisplay(cv::Mat frame);
void mode();

cv::String face_cascade_name = "C:/haarcascade_frontalface_alt.xml";
cv::String eye_cascade_name = "C:/cascade_cuda/haarcascade_lefteye_2splits.xml";

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;

std::string main_window_name = "Capture - Face detection";
std::string alarm = "alarm";

cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat showImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

int main(int, char)
{
	cv::Mat frame;
	clock_t time;

	// Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
	//if (!eye_cascade.load(eye_cascade_name)){ printf("--(!)Error loading eye cascade, please change eye_cascade_name in source code.\n"); return -1; };

	cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);


	//createCornerKernels();
	ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
		43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

	cv::VideoCapture cap(0); // open the default camera
	
	if (cap.isOpened()){

		/*printf("Initiation: Please setting blink input value");*/
		mode();
		std::cin >> blinkinput;
		blinkinput = 1;
		//printf("blinkinput:%d");

		while (true){
			//���� �ð��� count -> �׳� �� ���������� ����.. ���� �ð����� ����.
			time = clock();
			current = (int)(time / (double)1000);

			//1frame�� �� 0.2�� 
			framecount++;

			//printf("frame:%d\t current:%d\n", framecount,current);
			
			cap.read(frame);

			//flipping around y-axis.
			cv::flip(frame, frame, 1);
			frame.copyTo(debugImage);
			frame.copyTo(showImage);

			//call image processing function 
			if (!frame.empty()){
				detectAndDisplay(frame);
			}
			else{
				printf("No capture. please check out your cam");
			}

			//display current video. �������� ����
			imshow(main_window_name, showImage);

			//option 
			int c = cv::waitKey(10);
			if ((char)c == 'c')
			{
				printf("Total blink%d\n",blinkcount);
				break;
			}

			if ((char)c == 's'){
				
				std::cin.clear();
				std::cin.ignore();
			

				printf("resetting: Please resetting blink input value:");
				std::cin >> blinkinput;

			}

			frame.release();
			debugImage.release();

		}
	}

	//releaseCornerKernels();
	return 0;
}


void warning(){
	IplImage *image = cvLoadImage("C:/blinkwarning.jpg");

	cvNamedWindow("Warning", 1);
	cvShowImage("Warning", image);
	cvWaitKey(1000);

	cvReleaseImage(&image);
	cvDestroyWindow("Warning");
}

void mode(){
	IplImage *image = cvLoadImage("C:/mode.jpg");

	cvNamedWindow("Mode", 1);
	cvShowImage("Mode", image);
	cvWaitKey(1000);

	cvReleaseImage(&image);
	cvDestroyWindow("Mode");
}

void movewarning(){
	IplImage *image = cvLoadImage("C:/movewarning.jpg");

	cvNamedWindow("warning!", 1);
	cvShowImage("warning!", image);
	cvWaitKey(1000);

	cvReleaseImage(&image);
	cvDestroyWindow("warning!");
}

void blinkmode(int input){

	//1frame�� �� 0.2�� 
	if (current % (5*input) == 0)
	{
		if ((blinkcount - preblinkcount) > 1)
		{
			warning();
			printf("warning!\n");
		}
		preblinkcount = blinkcount;
	}

}

void findEyes(cv::Mat frame_gray, cv::Rect face){
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;

	if (kSmoothFaceImage){
		double sigma = kSmoothFaceFactor*face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}

	//find the eye region
	int eye_region_width = face.width*(kEyePercentWidth / 100.0);
	//�� �ڵ忡�� face.weight->������ Ȯ��
	int eye_region_height = face.height*(kEyePercentHeight / 100.0);
	int eye_region_top = face.height*(kEyePercentTop / 100.0);

	cv::Rect eyeregion(face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width,
		eye_region_height);
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width,
		eye_region_height);
	cv::Rect rightEyeRegion(face.width-eye_region_width-face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width,
		eye_region_height);

	//find eye center
	cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "LeftEye");
	cv::Point rightPupil = findEyeCenter(faceROI, leftEyeRegion, "RightEye");

	//change eye centers to face coord
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;

	circle(showImage, leftPupil, 1,1234,1,8,0);

	llx = lx;
	lly = ly;

	lx = leftPupil.x;
	ly = leftPupil.y;

	//printf("\nlx:%d", lx);
	if (abs(llx - lx) > face.width*0.2 || abs(lly - ly) > face.height*0.2 )
		movewarning();

	faceROI.release();
	debugFace.release();

}

//������ ��ġ�� �Ǻθ� ������ blinkcount�� �����ϴ� �Լ� 
cv::Mat findSkin(cv::Mat &frame){
	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows, frame.cols, CV_8U);

	cvtColor(frame, input, CV_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y){
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x){
			cv::Vec3b ycrcb = Mr[x];

			if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0){
				Or[x] = cv::Vec3b(0, 0, 0);
				skinpixel++;
			}
		}
	}

	//������ �߾� ��ġ ���� skin�� ����Ǿ����� �ƴ��� üũ 
	const cv::Vec3b *tr = input.ptr<cv::Vec3b>(lly);
	cv::Vec3b *er = frame.ptr<cv::Vec3b>(lly);
	cv::Vec3b tycrcb = tr[llx];

	//skin�� ���� ���� �ʾ��� ��� ���� �� ����. 
	if (skinCrCbHist.at<uchar>(tycrcb[1], tycrcb[2]) == 0) {

		//if (preskinpixel>skinpixel){
			printf("open\n");
			flag = 1;
			//printf("skinpixel:%d\n", skinpixel);
			printf("blink%d\n", blinkcount);
			//printf("move%d\n", motioncount);
		//}
			

	}
	//skin�� ���� �Ǿ��� ��� ���� ���� ����.
	else
	{
		//if (skinpixel<preskinpixel){
		if (flag == 1){
			motioncount++;
			blinkcount++;
			flag = 0;
			printf("closed\n");
			//printf("skinpixel:%d\n", skinpixel);
			printf("blink%d\n", blinkcount);
			printf("move%d\n", motioncount);

		}
		
		//}
	

	}


	preskinpixel = skinpixel;
	input.release();
	return output;

}

void detectAndDisplay(cv::Mat frame) {
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eye;
	
	std::vector<cv::Mat> rgbChannels(3);	//vector array rgbChannels����
	cv::split(frame, rgbChannels);			//frame�� rgbChannels�� reallocate?
	cv::Mat frame_gray = rgbChannels[2];    //n dimensional array class Mat. 0->1->2�� ������ ��� ���� ��.  grayscale�� �̾Ƴ��� ��.


	//-- Detect faces. ������ ��ü list of rectangles�� ��ȯ 
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

	//-- Show what you got
	if (faces.size() > 0) {
		findEyes(frame_gray, faces[0]);
		
		findSkin(debugImage(faces[0]));


	}

	blinkmode(blinkinput);

	frame_gray.release();
}





