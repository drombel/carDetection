#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

VideoCapture capture;
Mat frame, fgMaskMOG2, frame_copy;
Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();
RNG rng(12345);
int keyboard;
int history_min = 160, history_max_value = 300;
int nmixtures_min = 5, nmixtures_max_value = 7;
int carWidth, maxCarWidth = 200, carHeight, maxCarHeight = 200;
int checkLineY, carCounter, isDetectionOn = 1;

void BackgroundSubtractorProcess();
void detectCarsProcess();

void BackgroundSubtractorProcessChange(int, void*) {
	BackgroundSubtractorProcess();
}

void detectCarsProcessChange(int, void*) {
	detectCarsProcess();
}

void example1() {
	capture = VideoCapture("D:/videoCars-1.mp4");
	checkLineY = 200;
	carWidth = 50;
	carHeight = 70;
}

void example2() {
	capture = VideoCapture("D:/videoCars-2.mp4");
	checkLineY = 200;
	carWidth = 30;
	carHeight = 50;
}

void example3() {
	capture = VideoCapture("D:/videoCars-3.mp4");
	checkLineY = 200;
	carWidth = 25;
	carHeight = 40;
}

void example4() {
	capture = VideoCapture("C:/Users/Dawid/Downloads/videoCars-4.mp4");
	checkLineY = 200;
	carWidth = 25;
	carHeight = 40;
}

int main(int argc, char* argv[])
{
	namedWindow("Frame");
	namedWindow("Controls");
	namedWindow("FG Mask MOG 2");

	//VideoCapture capture(0);
	example1();
	//example2();
	//example3();
	//example4();

	createTrackbar("history", "Controls", &history_min, history_max_value, BackgroundSubtractorProcessChange);
	createTrackbar("min car width", "Controls", &carWidth, maxCarWidth, detectCarsProcessChange);
	createTrackbar("min car height", "Controls", &carHeight, maxCarHeight, detectCarsProcessChange);
	createTrackbar("off / On", "Controls", &isDetectionOn, 1, detectCarsProcessChange);
	//createTrackbar("nmixtures", "Controls", &nmixtures_min, nmixtures_max_value, BackgroundSubtractorProcessChange);

	if (!capture.isOpened()) {
		exit(EXIT_FAILURE);
	}

	pMOG2->setDetectShadows(true);
	//pMOG2->setComplexityReductionThreshold(16);
	pMOG2->setShadowValue(THRESH_BINARY);

	while (capture.read(frame)) {
		resize(frame, frame, Size(480, 360), 0, 0, INTER_CUBIC);
		frame_copy = frame.clone();

		BackgroundSubtractorProcess();
		if (isDetectionOn == 1)
			detectCarsProcess();

		imshow("Frame", frame_copy);

		int k = waitKey(30);
		if (k == 27) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
	}
	capture.release();
	destroyAllWindows();
	return EXIT_SUCCESS;
}

void BackgroundSubtractorProcess() {
	pMOG2->setHistory(history_min);
	if (nmixtures_min <= 5 && nmixtures_min >= 4)
		pMOG2->setNMixtures(nmixtures_min);

	pMOG2->apply(frame, fgMaskMOG2);

	imshow("FG Mask MOG 2", fgMaskMOG2);
}

void detectCarsProcess() {
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(fgMaskMOG2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// filter objects
	vector<Rect> rectList;
	for (int i = 0; i< contours.size(); i++){
		Rect rect = boundingRect(contours[i]);
		if (rect.width>carWidth && rect.height>carHeight)
			if ((rect.y + rect.height) >= checkLineY && checkLineY > rect.y)
				rectList.push_back(rect);
	}

	/// Draw rectangles
	for (int i = 0; i < rectList.size(); i++) {
		rectangle(frame_copy, rectList[i], Scalar(0, 255, 0), 3);
		cout << "width = " + to_string(rectList[i].width) + ", height = "+to_string(rectList[i].height) << endl;
	}

	putText(frame_copy, "Liczba samochodow: "+to_string(rectList.size()), Point(10, 20), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 2);
	line(frame_copy, Point(0, checkLineY), Point(480, checkLineY), Scalar(255, 0, 0), 4);
}