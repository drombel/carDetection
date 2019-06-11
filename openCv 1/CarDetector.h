#include "Blob.h"

#pragma once
namespace CarDetector
{
	void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
	void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
	void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
	double distanceBetweenPoints(cv::Point point1, cv::Point point2);
	void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
	void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
	bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &checkLineY, int &carCount);
	void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
	void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
};

