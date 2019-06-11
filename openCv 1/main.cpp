#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include "Blob.h"
#include "CarDetector.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace cv;
using namespace std;
using namespace CarDetector;

cv::VideoCapture example1() {
	return VideoCapture("../videoCars-1.mp4");
}

cv::VideoCapture example2() {
	return VideoCapture("../videoCars-2.mp4");
}

int main(int argc, char* argv[])
{
	//cvui::init("Frame");

	std::vector<Blob> blobs;
	cv::Point crossingLine[2];
	Mat frame1, frame2;
	VideoCapture capture;
	int carCount = 0, checkLineY = 200;

	// controlls
	cv::Mat controllerFrame = cv::Mat(480, 360, CV_8UC3);
	string WINDOW_NAME = "Controller Frame";
	cvui::init(WINDOW_NAME);
	int structuingValueIndex = 5, blurIndex = 5, widthCarIndex = 60, heightCarIndex = 60;
	bool carDetectionOn;

	//VideoCapture capture(0);
	capture = example1();
	//capture = example2();

	if (!capture.isOpened() || capture.get(CAP_PROP_FRAME_COUNT) < 2)
		exit(EXIT_FAILURE);

	capture.read(frame1);
	resize(frame1, frame1, Size(480, 360), 0, 0, INTER_CUBIC);

	crossingLine[0] = Point(0, checkLineY);
	crossingLine[1] = Point(frame1.cols - 1, checkLineY);

	bool blnFirstFrame = true;

	while (capture.read(frame2)) {
		resize(frame2, frame2, Size(480, 360), 0, 0, INTER_CUBIC);

		controllerFrame = cv::Scalar(49, 52, 49);		
		
		cvui::checkbox(controllerFrame, 20, 50, "On / Off", &carDetectionOn);
		cvui::text(controllerFrame, 20, 80, "Wybierz rozmiar filtracji", 0.4, 0xff0000);
		cvui::trackbar(controllerFrame, 20, 110, 150, &structuingValueIndex, 1, 12);
		cvui::text(controllerFrame, 20, 160, "Blur", 0.4, 0xff0000);
		cvui::trackbar(controllerFrame, 20, 190, 150, &blurIndex, 1, 7);
		cvui::text(controllerFrame, 20, 240, "Szerkosc auta", 0.4, 0xff0000);
		cvui::trackbar(controllerFrame, 20, 270, 150, &widthCarIndex, 20, 150);
		cvui::text(controllerFrame, 20, 320, "Wysokosc auta", 0.4, 0xff0000);
		cvui::trackbar(controllerFrame, 20, 350, 150, &heightCarIndex, 20, 150);

		cvui::update();
		cv::imshow(WINDOW_NAME, controllerFrame);

		int k = waitKey(25);
		if (k == 27) exit(EXIT_FAILURE);

		if (!carDetectionOn) {
			cv::imshow("frame2_copy", frame2);
			continue;
		}

		std::vector<Blob> currentFrameBlobs;

		cv::Mat 
			frame1_copy = frame1.clone(),
			frame2_copy = frame2.clone(),
			frame_diff, frame_thresh;

		cv::cvtColor(frame1_copy, frame1_copy, COLOR_BGR2GRAY);
		cv::cvtColor(frame2_copy, frame2_copy, COLOR_BGR2GRAY);

		if (blurIndex % 2 == 0)
			++blurIndex;

		cv::GaussianBlur(frame1_copy, frame1_copy, cv::Size(blurIndex, blurIndex), 0);
		cv::GaussianBlur(frame2_copy, frame2_copy, cv::Size(blurIndex, blurIndex), 0);

		// roznica pomiedzy poprzednia klatka, a aktualna
		cv::absdiff(frame1_copy, frame2_copy, frame_diff);

		// binaryzacja
		cv::threshold(frame_diff, frame_thresh, 30, 255.0, THRESH_BINARY);

		//cv::imshow("frame_thresh", frame_thresh);

		cv::Mat currentStuctElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(structuingValueIndex, structuingValueIndex));

		for (int i = 0; i < 2; i++) {
			cv::dilate(frame_thresh, frame_thresh, currentStuctElement, Point(-1, -1), 2);
			cv::erode(frame_thresh, frame_thresh, currentStuctElement);
		}

		cv::Mat frame_thresh_copy = frame_thresh.clone();

		std::vector<std::vector<cv::Point>> contours;

		cv::findContours(frame_thresh_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//drawAndShowContours(frame_thresh.size(), contours, "imgContours");

		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (int i = 0; i < contours.size(); i++)
			cv::convexHull(contours[i], convexHulls[i]);

		//drawAndShowContours(frame_thresh.size(), convexHulls, "imgConvexHulls");

		for (auto &convexHull : convexHulls) {
			Blob blob_params(convexHull);

			if (
				blob_params.dblCurrentAspectRatio > 0.2 &&
				blob_params.dblCurrentAspectRatio < 4.0 &&
				blob_params.currentBoundingRect.width > widthCarIndex &&
				blob_params.currentBoundingRect.height > heightCarIndex
				)
				currentFrameBlobs.push_back(blob_params);
		}

		drawAndShowContours(frame_thresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true) 
			for (auto &currentFrameBlob : currentFrameBlobs) 
				blobs.push_back(currentFrameBlob);
		else
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);


		drawAndShowContours(frame_thresh.size(), blobs, "imgBlobs");

		frame2_copy = frame2.clone();          
		drawBlobInfoOnImage(blobs, frame2_copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, checkLineY, carCount);

		if (blnAtLeastOneBlobCrossedTheLine)
			cv::line(frame2_copy, crossingLine[0], crossingLine[1], Scalar(0, 255, 0), 2);
		else
			cv::line(frame2_copy, crossingLine[0], crossingLine[1], Scalar(0, 0, 255), 2);
		
		drawCarCountOnImage(carCount, frame2_copy);
		cv::imshow("frame2_copy", frame2_copy);

		currentFrameBlobs.clear();
		frame1 = frame2.clone();
		blnFirstFrame = false;

		
	}

	capture.release();
	cv::destroyAllWindows();
	return EXIT_SUCCESS;
}
