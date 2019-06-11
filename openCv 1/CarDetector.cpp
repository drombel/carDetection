#include "CarDetector.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace CarDetector{
	void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

		for (auto &existingBlob : existingBlobs) {
			existingBlob.blnCurrentMatchFoundOrNewBlob = false;
			existingBlob.predictNextPosition();
		}

		for (auto &currentFrameBlob : currentFrameBlobs) {

			int intIndexOfLeastDistance = 0;
			double dblLeastDistance = 100000.0;

			for (int i = 0; i < existingBlobs.size(); i++) {

				if (existingBlobs[i].blnStillBeingTracked == true) {

					double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

					if (dblDistance < dblLeastDistance) {
						dblLeastDistance = dblDistance;
						intIndexOfLeastDistance = i;
					}
				}
			}

			if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
				addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
			}
			else {
				addNewBlob(currentFrameBlob, existingBlobs);
			}

		}

		for (auto &existingBlob : existingBlobs) {

			if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
				existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
			}

			if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 10) {
				existingBlob.blnStillBeingTracked = false;
			}

		}

	}

	void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

		existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
		existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

		existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

		existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
		existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

		existingBlobs[intIndex].blnStillBeingTracked = true;
		existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
	}

	void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

		currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

		existingBlobs.push_back(currentFrameBlob);
	}

	double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

		int intX = abs(point1.x - point2.x);
		int intY = abs(point1.y - point2.y);

		return(sqrt(pow(intX, 2) + pow(intY, 2)));
	}

	void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
		cv::Mat image(imageSize, CV_8UC3, Scalar(0, 0, 0));

		cv::drawContours(image, contours, -1, Scalar(255, 255, 255), -1);

		cv::imshow(strImageName, image);
	}

	void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

		cv::Mat image(imageSize, CV_8UC3, Scalar(0, 0, 0));

		std::vector<std::vector<cv::Point> > contours;

		for (auto &blob : blobs)
			if (blob.blnStillBeingTracked == true)
				contours.push_back(blob.currentContour);

		cv::drawContours(image, contours, -1, Scalar(255, 255, 255), -1);
		cv::imshow(strImageName, image);
	}

	bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &checkLineY, int &carCount) {
		bool blnAtLeastOneBlobCrossedTheLine = false;

		//cout << endl;
		for (auto &blob : blobs) {

			if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
				int prevFrameIndex = (int)blob.centerPositions.size() - 2;
				int currFrameIndex = (int)blob.centerPositions.size() - 1;

				//cout << "prev = " + to_string(blob.centerPositions[prevFrameIndex].y) + " > " + to_string(checkLineY) + ", curr = " + to_string(blob.centerPositions[currFrameIndex].y) + " <= " + to_string(checkLineY) << endl;
				bool crossedLine, movingUp = (blob.centerPositions[prevFrameIndex].y - blob.centerPositions[currFrameIndex].y) >= 0;
				if (movingUp)
					crossedLine = blob.centerPositions[prevFrameIndex].y > checkLineY && blob.centerPositions[currFrameIndex].y <= checkLineY;
				else
					crossedLine = blob.centerPositions[prevFrameIndex].y <= checkLineY && blob.centerPositions[currFrameIndex].y > checkLineY;

				if (crossedLine) {
					carCount++;
					cout << to_string(carCount) << endl;
					blob.blnStillBeingTracked = false;
					blnAtLeastOneBlobCrossedTheLine = true;
				}
			}

		}

		return blnAtLeastOneBlobCrossedTheLine;
	}

	void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {
		for (int i = 0; i < blobs.size(); i++)
			if (blobs[i].blnStillBeingTracked)
				cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, Scalar(255, 0, 0), 2);

	}

	void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {
		putText(imgFrame2Copy, "Liczba samochodow: " + to_string(carCount), Point(10, 40), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 2);
	}
}