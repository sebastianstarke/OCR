
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

# define M_PI           3.14159265358979323846 

enum projectionID{VERTICAL,HORIZONTAL};

//Structs
struct Interval
{
	int begin;
	int length;
};
struct Letter
{
	Mat image;
	Rect boundingRect;
	vector<double> featureVector;
	int label;
};
struct Manuscript
{
	string fileName;
	Mat image;
	vector<Letter> letters;
	int author;
};

// Function Headers
void setIntensity(Mat &image, int x, int y, int intensity);
int getIntensity(Vec3b pixel);
Vec3b getPixel(Mat &image, int x, int y);
void setPixel(Mat &image, int x, int y, Vec3b pixel);

//Detection
void binaryActivityMask(Mat &input, Mat &output, int width, int height, int minActivity);
void binaryNeighbourhoodMask(Mat &input, Mat &output, int minActivity);
void intensityThresholdFilter(Mat &input, Mat &output, int minIntensity, int maxIntensity, bool binarize);
vector<Interval> intensityProjectionFilter(Mat &input, Mat &output, int projectionID, int xMin, int xMax, int yMin, int yMax, int mapToZeroThreshold, int minLength);
vector<Mat> extractImages(Mat &image, std::vector<Rect> boundingRects);
void detectChineseLetters(Manuscript &manuscript, bool showProcess, bool showResult);

double intensityFluctuations(Mat &input, int projectionID);

//Features
vector<double> computeFeatureVector(Mat &image);
double averageIntensity(Mat &image);
double intensityVariance(Mat &image);
double averageGradient(Mat &image);
double boundingRectFilling(Mat &image);
double aspectRatio(Mat &image);
double regions(Mat &image);
double avgRegionSize(Mat &image);
double gradientVariance(Mat &image);
double lineThickness(Mat &image);
double lineThicknessVariance(Mat &image);
double averageCurvature(Mat &image);
double varianceCurvature(Mat &image);
double letterSize(Mat &image);

void normalizeDimension(vector<Letter> &letters, int dim);
double getMinFeatureValue(vector<double> &featureDimension);
double getMaxFeatureValue(vector<double> &featureDimension);

//DRAWING
void plotFeatureDistribution(string name, vector<Letter> &letters, int dim);
void plotPoints(string name, int width, int height, vector<Point>);
int plotNr = 1;

//Classify
void sortDistancesAndLabels(vector<double> &distances, vector<int> &labels, int p, int q);
int partition(vector<double> &distances, vector<int> &labels, int p, int q);

//Branches
void processManuscripts(string file1, string file2, int k, double sigma, bool visualization);
void minimizeTrainingError(int kMin, int kMax, int kStep, double sigmaMin, double sigmaMax, double sigmaStep, bool errorOnLetters);

// Function main
void main(int argc, char* argv[])
{
	/* PROGRAM */
	
	if (argc - 1 == 5) {
		string file1 = argv[1];
		string file2 = argv[2];
		int k = atoi(argv[3]); //default 100
		double sigma = stod(argv[4]); //default 0.25
		bool visualization = atoi(argv[5]);
		processManuscripts(file1, file2, k, sigma, visualization);
	}
	else {
		cout << "Input: [FILE1] [FILE2] [K] [SIGMA] [VISUALIZATION]" << endl;
	}
	

	/* TRAINING ERROR */
	/*
	if (argc - 1 == 7) {
		int kMin = atoi(argv[1]);
		int kMax = atoi(argv[2]);
		int kStep = atoi(argv[3]);
		double sigmaMin = stod(argv[4]);
		double sigmaMax = stod(argv[5]);
		double sigmaStep = stod(argv[6]);
		bool errorOnLetters = atoi(argv[7]);
		minimizeTrainingError(kMin, kMax, kStep, sigmaMin, sigmaMax, sigmaStep, errorOnLetters);
	}
	else {
		cout << "Input: [KMIN] [KMAX] [KSTEP] [SIGMAMIN] [SIGMAMAX] [SIGMASTEP] [ERROR ON LETTERS]" << endl;
	}
	*/

	//Wait until end
	for (;;)
	{
		char key = (char)waitKey(20);
		if (key == 27)
			break;
	}
}

void processManuscripts(string file1, string file2, int k, double sigma, bool visualization) {
	Manuscript manuscript_1;
	Manuscript manuscript_2;

	manuscript_1.fileName = file1;
	manuscript_1.image = imread(manuscript_1.fileName);
	manuscript_1.author = 1;
	detectChineseLetters(manuscript_1, visualization, true);
	cout << "Computing feature vectors ...";
	for (int i = 0; i<manuscript_1.letters.size(); i++)
	{
		manuscript_1.letters.at(i).featureVector = computeFeatureVector(manuscript_1.letters.at(i).image);
	}
	cout << " Done!" << endl << endl;

	manuscript_2.fileName = file2;
	manuscript_2.image = imread(manuscript_2.fileName);
	manuscript_2.author = 2;
	detectChineseLetters(manuscript_2, visualization, true);
	cout << "Computing feature vectors ...";
	for (int i = 0; i<manuscript_2.letters.size(); i++)
	{
		manuscript_2.letters.at(i).featureVector = computeFeatureVector(manuscript_2.letters.at(i).image);
	}
	cout << " Done!" << endl << endl;

	//Gather all letters
	vector<Letter> letters;
	for (int i = 0; i<manuscript_1.letters.size(); i++)
	{
		letters.push_back(manuscript_1.letters.at(i));
	}
	for (int i = 0; i<manuscript_2.letters.size(); i++)
	{
		letters.push_back(manuscript_2.letters.at(i));
	}

	//Normalize Feature Vectors Dimensions
	normalizeDimension(letters, 0);
	normalizeDimension(letters, 1);
	normalizeDimension(letters, 2);
	normalizeDimension(letters, 3);
	normalizeDimension(letters, 4);
	normalizeDimension(letters, 5);
	normalizeDimension(letters, 6);
	normalizeDimension(letters, 7);
	normalizeDimension(letters, 8);
	normalizeDimension(letters, 9);
	normalizeDimension(letters, 10);
	normalizeDimension(letters, 11);
	normalizeDimension(letters, 12);

	//Plot Distributions
	if (visualization)
	{
		plotFeatureDistribution("Intensity (Average)", letters, 0);
		plotFeatureDistribution("Intensity (Variance)", letters, 1);
		plotFeatureDistribution("Gradient (Average)", letters, 2);
		plotFeatureDistribution("Gradient (Variance)", letters, 3);
		plotFeatureDistribution("Curvature (Average)", letters, 4);
		plotFeatureDistribution("Curvature (Variance)", letters, 5);
		plotFeatureDistribution("Letter-Size", letters, 6);
		plotFeatureDistribution("Segments (Count)", letters, 7);
		plotFeatureDistribution("Segments (Size)", letters, 8);
		plotFeatureDistribution("Filling", letters, 9);
		plotFeatureDistribution("Aspect-Ratio", letters, 10);
		plotFeatureDistribution("Line-Thickness (Average)", letters, 11);
		plotFeatureDistribution("Line-Thickness (Variance)", letters, 12);
	}

	//kNN Classification
	cout << "Running kNN-Classification with Parameters:" << endl << "k = " << k << endl << "sigma = " << sigma << endl;

	Mat letterImage_c1 = manuscript_1.image.clone();
	Mat letterImage_c2 = manuscript_2.image.clone();

	int decisionCount = 0;

	for (int pos = 0; pos<letters.size(); pos++)
	{

		vector<vector<double>> featureVectors;
		vector<int> labels;
		for (int i = 0; i<letters.size(); i++)
		{
			featureVectors.push_back(letters.at(i).featureVector);
			labels.push_back(letters.at(i).label);
		}

		vector<double> distances;
		for (int i = 0; i<featureVectors.size(); i++)
		{
			double distance = 0;
			for (int j = 0; j<featureVectors.at(i).size(); j++)
			{
				distance += pow(featureVectors.at(i).at(j) - letters.at(pos).featureVector.at(j), 2);
			}
			distance = sqrt(distance);
			distances.push_back(distance);
		}
		sortDistancesAndLabels(distances, labels, 0, distances.size());

		vector<int> predictionSet;
		for (int i = 1; i<k + 1; i++)
		{
			predictionSet.push_back(labels.at(i));
		}

		int count_c1 = 0;
		int count_c2 = 0;
		for (int i = 0; i<predictionSet.size(); i++)
		{
			if (predictionSet.at(i) == 1)
				count_c1++;
			if (predictionSet.at(i) == 2)
				count_c2++;
		}

		int maxCount = 0;
		if (count_c1 > count_c2)
		{
			maxCount = count_c1;
		}
		if (count_c2 > count_c1)
		{
			maxCount = count_c2;
		}
		if (count_c1 == count_c2)
		{
			maxCount = count_c1;
		}

		double confidence = double(double(maxCount) / double(predictionSet.size()));
		if (confidence >= 0.5 - sigma && confidence <= 0.5 + sigma)
		{
			decisionCount++;
			if (letters.at(pos).label == 1)
			{
				rectangle(letterImage_c1, letters.at(pos).boundingRect, Scalar(0, 255, 0), 2);
			}
			else
			{
				rectangle(letterImage_c2, letters.at(pos).boundingRect, Scalar(0, 255, 0), 2);
			}
		}

	}
	double confidence = double(double(decisionCount) / double(letters.size()));
	cout << "=================================================================" << endl;
	cout << "Confidence of the manuscripts being written by the same author: " << endl << confidence << " (" << decisionCount << " / " << letters.size() << " matching letters found)" << endl;
	cout << "-----------------------------------------------------------------" << endl;
	cout << "Conclusion: ";
	string result;
	if (confidence == 0.0)
		result = "Impossible!";
	else if (confidence < 0.2)
		result = "Very unlikely!";
	else if (confidence < 0.4)
		result = "Rather unlikely!";
	else if (confidence < 0.5)
		result = "Not sure, but rather unlikely...";
	else if (confidence < 0.6)
		result = "Not sure, but rather likely...";
	else if (confidence < 0.8)
		result = "Rather likely!";
	else if (confidence < 1.0)
		result = "Very likely!";
	else if (confidence == 1.0)
		result = "Completely sure!";
	cout << result << endl;
	cout << "=================================================================" << endl;

	resize(letterImage_c1, letterImage_c1, Size(letterImage_c1.cols*0.75, letterImage_c1.rows*0.75));
	resize(letterImage_c2, letterImage_c2, Size(letterImage_c2.cols*0.75, letterImage_c2.rows*0.75));

	imshow(manuscript_1.fileName + " (Matching Letters)", letterImage_c1);
	imshow(manuscript_2.fileName + " (Matching Letters)", letterImage_c2);
}

void detectChineseLetters(Manuscript &manuscript, bool showProcess, bool showResult)
{
	cout << "Detecting letters in " << manuscript.fileName << endl;

	if(showProcess)
		imshow(manuscript.fileName + " (Original)", manuscript.image);

	Mat processingImage = manuscript.image.clone();
	Mat morphologyImage, intensityThresholdImage, intensityProjectionImage;
	Mat detectionImage = manuscript.image.clone();

	//Apply Morphology Filter to exhibit important Regions
	int morph_elem = 1;
	int morph_size = 16;
	int morph_operator = 4;
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	morphologyEx(processingImage, morphologyImage, morph_operator+2, element);
	if(showProcess)
		imshow(manuscript.fileName + " (Morphology)", morphologyImage);

	//Intensity Threshold filter
	intensityThresholdFilter(morphologyImage,intensityThresholdImage,50,255,true);
	if(showProcess)
		imshow(manuscript.fileName + " (Intensity-Threshold)", intensityThresholdImage);

	//Noise Filter by binary activity mask // TODO!
	binaryNeighbourhoodMask(intensityThresholdImage,intensityThresholdImage,2);
	//binaryActivityMask(processingImage,processingImage,10,10,10);
	if(showProcess)
		imshow(manuscript.fileName + " (kNN-Noise-Filter)", intensityThresholdImage);

	//Perform Intensity Projection to seperate vertically and horizontally
	int mapToZeroThreshold = 15;
	int minLength = 15;
	std::vector<Rect> boundingRects;
	std::vector<Interval> verticalRegions = intensityProjectionFilter(intensityThresholdImage,intensityProjectionImage,VERTICAL,0,intensityThresholdImage.cols,0,intensityThresholdImage.rows,mapToZeroThreshold,minLength);
	for(int i=0; i<verticalRegions.size(); i++)
	{
		std::vector<Interval> horizontalRegions = intensityProjectionFilter(intensityProjectionImage,intensityProjectionImage,HORIZONTAL,verticalRegions.at(i).begin,verticalRegions.at(i).begin+verticalRegions.at(i).length,0,intensityThresholdImage.rows,mapToZeroThreshold,minLength);
		for(int j=0; j<horizontalRegions.size(); j++)
		{
			double expandFactor = 0.3;
			double x = verticalRegions.at(i).begin;
			double y = horizontalRegions.at(j).begin;
			double width = verticalRegions.at(i).length;
			double height = horizontalRegions.at(j).length;
			
			double expandX = width*expandFactor;
			double expandY = height*expandFactor;
			width += expandY;
			height += expandX;
			
			x -= 0.5*expandY;
			if(x<0)
				x=0;
			if(x>manuscript.image.cols)
				x=manuscript.image.cols-1-width;
			y -= 0.5*expandX;
			if(y<0)
				y=0;
			if(y>manuscript.image.rows)
				y=manuscript.image.rows-1-height;

			boundingRects.push_back(Rect(x,y,width,height));
		}
	}
	if(showProcess)
		imshow(manuscript.fileName + " (Intensity-Projection)",intensityProjectionImage);

	//Extract Letter Images (Filter false positives with low fluctuations (borders) to obtain a better training set)
	std::vector<Mat> chineseLetterImages;
	chineseLetterImages = extractImages(intensityThresholdImage,boundingRects);
	for(int i=0; i<chineseLetterImages.size(); i++)
	{
		vector<Point> nonZeroPixels;
		for(int x=0; x<chineseLetterImages.at(i).cols; x++)
		{
			for(int y=0; y<chineseLetterImages.at(i).rows; y++)
			{
				if(getIntensity(getPixel(chineseLetterImages.at(i),x,y)) != 0)
				{
					nonZeroPixels.push_back(Point(boundingRects.at(i).x,boundingRects.at(i).y)+Point(x,y));
				}
			}
		}
		boundingRects.at(i) = boundingRect(nonZeroPixels);

		if(intensityFluctuations(chineseLetterImages.at(i),VERTICAL) <= 1 || intensityFluctuations(chineseLetterImages.at(i),HORIZONTAL) <= 1)
		{
			chineseLetterImages.erase(chineseLetterImages.begin() + i);
			boundingRects.erase(boundingRects.begin() + i);
			i--;
		}
	}
	chineseLetterImages = extractImages(manuscript.image,boundingRects);

	//Visualize Bounding Boxes
	Scalar color;
		if(manuscript.author == 1)
			color = Scalar(255,0,0);
		if(manuscript.author == 2)
			color = Scalar(0,0,255);
	for(int i=0; i<chineseLetterImages.size(); i++)
	{
		rectangle(detectionImage,boundingRects.at(i),color,2);
	}
	if(showResult)
		imshow(manuscript.fileName + " (Detection Image)", detectionImage);

	//Create Letters
	for(int i=0; i<chineseLetterImages.size(); i++)
	{
		Letter letter;
		letter.image = chineseLetterImages.at(i);
		letter.boundingRect = boundingRects.at(i);
		letter.label = manuscript.author;
		manuscript.letters.push_back(letter);
	}

	cout << "Detection found " << manuscript.letters.size() << " chinese symbols in a manuscript of author " << manuscript.author << "!" << endl;
}

Vec3b getPixel(Mat &image, int x, int y)
{
	Vec3b* pixel = &image.at<Vec3b>(Point(x,y));
	return Vec3b(pixel->val[2],pixel->val[1],pixel->val[0]);
}

void setPixel(Mat &image, int x, int y, Vec3b pixel)
{
	image.at<Vec3b>(Point(x,y)) = Vec3b(pixel.val[2],pixel.val[1],pixel.val[0]);
}

int getIntensity(Vec3b pixel)
{
	return int((pixel.val[0]+pixel.val[1]+pixel.val[2])/3.0);
}

void setIntensity(Mat &image, int x, int y, int intensity)
{
	setPixel(image,x,y,Vec3b(intensity,intensity,intensity));
}

void intensityThresholdFilter(Mat &input, Mat &output, int minIntensity, int maxIntensity, bool binarize)
{
	output = input.clone();
	for(int y=0; y<output.rows; y++)
	{
		for(int x=0; x<output.cols; x++)
		{
			int intensity = getIntensity(getPixel(output,x,y));
			if(intensity<minIntensity || intensity>maxIntensity)
			{
				setPixel(output,x,y,Vec3b(0,0,0));
			}
			else
			{
				if(binarize)
					setPixel(output,x,y,Vec3b(255,255,255));
			}
		}
	}
}

void binaryNeighbourhoodMask(Mat &input, Mat &output, int minActivity)
{
	output = input.clone();
	for(int x=0; x<input.cols; x++)
	{
		for(int y=0; y<input.rows; y++)
		{
			int activeNeighbors = 0;

			if(x-1 >= 0)
			{
				if(getIntensity(getPixel(input,x-1,y)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(x+1 < input.cols)
			{
				if(getIntensity(getPixel(input,x+1,y)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(y-1 >= 0)
			{
				if(getIntensity(getPixel(input,x,y-1)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(y+1 < input.rows)
			{
				if(getIntensity(getPixel(input,x,y+1)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(x-1 >= 0 && y-1 >= 0)
			{
				if(getIntensity(getPixel(input,x-1,y-1)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(x+1 < input.cols && y+1 < input.rows)
			{
				if(getIntensity(getPixel(input,x+1,y+1)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(x-1 >= 0 && y+1 < input.rows)
			{
				if(getIntensity(getPixel(input,x-1,y+1)) > 0)
				{
					activeNeighbors++;
				}
			}
			if(x+1 < input.cols && y-1 >= 0)
			{
				if(getIntensity(getPixel(input,x+1,y-1)) > 0)
				{
					activeNeighbors++;
				}
			}

			if(activeNeighbors < minActivity)
			{
				setPixel(output,x,y,Vec3b(0,0,0));
			}
		}
	}
}

void binaryActivityMask(Mat &input, Mat &output, int width, int height, int minActivity)
{
	output = input.clone();
	width = width-1;
	height = height-1;
	for(int x=0; x<input.cols-width; x++)
	{
		for(int y=0; y<input.rows-height; y++)
		{
			int activity = 0;
			for(int i=x; i<x+width; i++)
			{
				for(int j=y; j<y+height; j++)
				{
					if(getIntensity(getPixel(input,i,j)) > 0)
					{
						activity++;
					}
				}
			}
			if(activity < minActivity)
			{
				for(int i=x; i<x+width; i++)
				{
					for(int j=y; j<y+height; j++)
					{
						setPixel(output,x,y,Vec3b(0,0,0));
					}
				}
			}
			else
			{
				for(int i=x; i<x+width; i++)
				{
					for(int j=y; j<y+height; j++)
					{
						if(getIntensity(getPixel(input,i,j)) > 0)
						{
							setPixel(output,i,j,Vec3b(255,255,255));
						}
					}
				}
			}
		}
	}
}

std::vector<Interval> intensityProjectionFilter(Mat &input, Mat &output, int projectionID, int xMin, int xMax, int yMin, int yMax, int mapToZeroThreshold, int minLength)
{
	output = input.clone();

	std::vector<int> intensityProjection;
	std::vector<Interval> intervals;

	if(projectionID == VERTICAL)
	{
		//Compute average intensity at each row or column and erase if value below mapToZero threshold
		for(int x=xMin; x<xMax; x++)
		{
			int avgIntensity = 0;
			for(int y=yMin; y<yMax; y++)
			{
				avgIntensity += getIntensity(getPixel(output,x,y));
			}
			avgIntensity /= (yMax-yMin);
			//Apply Threshold Mapping
			if(avgIntensity <= mapToZeroThreshold)
			{
				avgIntensity = 0;
				line(output, Point(x,yMin), Point(x,yMax), cv::Scalar(0,0,0),1);
			}
			intensityProjection.push_back(avgIntensity);
		}
	
		//Get length of regions
		Interval interval; interval.begin = 0; interval.length = 0;
		bool scanning = false;
		for(int i=0; i<intensityProjection.size(); i++)
		{
			if(intensityProjection.at(i) != 0)
			{
				if(!scanning)
				{
					interval.begin = i;
					scanning = true;
				}
				interval.length++;
			}
			else
			{
				if(interval.length>=minLength)
				{
					intervals.push_back(interval);
				}
				else
				{
					for(int j=0; j<interval.length; j++)
					{
						line(output, Point(xMin+i-j-1,yMin), Point(xMin+i-j-1,yMax), cv::Scalar(0,0,0),1);
					}
				}
				interval.begin = 0;
				interval.length = 0;
				scanning = false;
			}
		}
	}

	if(projectionID == HORIZONTAL)
	{
		//Compute average intensity at each row or column and erase if value below mapToZero threshold
		for(int y=yMin; y<yMax; y++)
		{
			int avgIntensity = 0;
			for(int x=xMin; x<xMax; x++)
			{
				avgIntensity += getIntensity(getPixel(output,x,y));
			}
			avgIntensity /= (xMax-xMin);
			//Apply Threshold Mapping
			if(avgIntensity <= mapToZeroThreshold)
			{
				avgIntensity = 0;
				line(output, Point(xMin,y), Point(xMax,y), cv::Scalar(0,0,0),1);
			}
			intensityProjection.push_back(avgIntensity);
		}
	
		//Get length of regions
		Interval interval; interval.begin = 0; interval.length = 0;
		bool scanning = false;
		for(int i=0; i<intensityProjection.size(); i++)
		{
			if(intensityProjection.at(i) != 0)
			{
				if(!scanning)
				{
					interval.begin = i;
					scanning = true;
				}
				interval.length++;
			}
			else
			{
				if(interval.length>=minLength)
				{
					intervals.push_back(interval);
				}
				else
				{
					for(int j=0; j<interval.length; j++)
					{
						line(output, Point(xMin,yMin+i-j-1), Point(xMax,yMin+i-j-1), cv::Scalar(0,0,0),1);
					}
				}
				interval.begin = 0;
				interval.length = 0;
				scanning = false;
			}
		}
	}

	return intervals;
}

double intensityFluctuations(Mat &input, int projectionID)
{
	double fluctations = 0;
	if(projectionID == VERTICAL)
	{
		for(int x=0; x<input.cols; x++)
		{
			int intensity = getIntensity(getPixel(input,x,0));;
			for(int y=0; y<input.rows; y++)
			{
				if(intensity != getIntensity(getPixel(input,x,y)))
				{
					intensity = getIntensity(getPixel(input,x,y));
					fluctations++;
				}
			}
		}
		fluctations /= double(input.cols);
	}

	if(projectionID == HORIZONTAL)
	{
		for(int y=0; y<input.rows; y++)
		{
			int intensity = getIntensity(getPixel(input,0,y));;
			for(int x=0; x<input.cols; x++)
			{
				if(intensity != getIntensity(getPixel(input,x,y)))
				{
					intensity = getIntensity(getPixel(input,x,y));
					fluctations++;
				}
			}
		}
		fluctations /= double(input.rows);
	}
	return fluctations;
}

std::vector<Mat> extractImages(Mat &image, std::vector<Rect> boundingRects)
{
	std::vector<Mat> images;
	for(int i=0; i<boundingRects.size(); i++)
	{
		Mat subImage(image, boundingRects.at(i));
		images.push_back(subImage);
	}
	return images;
}

//FEATURES
vector<double> computeFeatureVector(Mat &image)
{
	Mat img = image.clone();
	vector<double> featureVector;

	//
	Mat imgThreshold;
	intensityThresholdFilter(img,imgThreshold,20,150,false);

	//Compute intensity average
	double avgIntensity = averageIntensity(imgThreshold);
	featureVector.push_back(avgIntensity);

	//Compute intensity variance
	double variance = intensityVariance(imgThreshold);
	featureVector.push_back(variance);

	//Compute gradient average
	double avgGradient = averageGradient(imgThreshold);
	featureVector.push_back(avgGradient);
	
	//Compute gradient variance
	double agf = gradientVariance(imgThreshold);
	featureVector.push_back(agf);
	
	//Compute curvature average
	double avgCurvature = averageCurvature(imgThreshold);
	featureVector.push_back(avgCurvature);

	//Compute curvature variance
	double varCurvature = varianceCurvature(imgThreshold);
	featureVector.push_back(varCurvature);

	//Compute lettersize average
	double size = letterSize(imgThreshold);
	featureVector.push_back(size);

	//Compute number of regions
	double nr = regions(imgThreshold);
	featureVector.push_back(nr);

	//Compute average region size
	double arg = avgRegionSize(imgThreshold);
	featureVector.push_back(arg);

	//Compute filling of bounding rectagnle
	double filling = boundingRectFilling(imgThreshold);
	featureVector.push_back(filling);

	//Compute aspect ratio
	double ar = aspectRatio(imgThreshold);
	featureVector.push_back(ar);

	//Compute average line thickness
	double thickness = lineThickness(imgThreshold);
	featureVector.push_back(thickness);

	//Compute variance line thickness
	double thicknessVariance = lineThicknessVariance(imgThreshold);
	featureVector.push_back(thicknessVariance);

	return featureVector;
}

double averageIntensity(Mat &image)
{
	double intensity = 0;
	int count = 0;
	for(int x=0; x<image.cols; x++)
	{
		for(int y=0; y<image.rows; y++)
		{
			double pixelIntensity = getIntensity(getPixel(image,x,y));
			if(pixelIntensity != 0)
			{
				intensity += pixelIntensity;
				count++;
			}
		}
	}
	intensity /= count;
	return intensity;
}

double intensityVariance(Mat &image)
{
	double sum = 0;
	double mean = averageIntensity(image);
	int count = 0;
	for(int x=0; x<image.cols; x++)
	{
		for(int y=0; y<image.rows; y++)
		{
			double pixelIntensity = getIntensity(getPixel(image,x,y));
			if(pixelIntensity != 0)
			{
				sum += pow(pixelIntensity-mean,2);
				count++;
			}
		}
	}
	return double(sum/count); 
}

double boundingRectFilling(Mat &image)
{
	Mat img = image.clone();

	int nonZeroPixels = 0;
	for(int x=0; x<img.cols; x++)
	{
		for(int y=0; y<img.rows; y++)
		{
			if(getIntensity(getPixel(img,x,y)) != 0)
			{
				nonZeroPixels++;
			}
		}
	}
	
	double boundingRectFilling = double(double(nonZeroPixels)/(double(img.rows)*double(img.cols))) * 1000;

	return boundingRectFilling;
}

double aspectRatio(Mat &image)
{
	return double(double(image.cols)/double(image.rows))*10;
}

double regions(Mat &image)
{	
	Mat img = image.clone();
	cvtColor(img, img, CV_BGR2GRAY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	return contours.size();
}

double avgRegionSize(Mat &image)
{
	Mat img = image.clone();
	cvtColor(img, img, CV_BGR2GRAY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	double avgRegionSize = 0;
	for( int i = 0; i< contours.size(); i++ )
	{
		avgRegionSize += contourArea(contours.at(i),false);
	}
	avgRegionSize /= contours.size();
	
	return avgRegionSize;
}

double averageGradient(Mat &image)
{
	Mat src_gray = image.clone();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			if(getIntensity(getPixel(src_gray,x,y)) != 0)
				setPixel(src_gray,x,y,Vec3b(255,255,255));
		}
	}

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;

	/// Gradient X
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

	/// Gradient Y
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	vector<double> gradients;
	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			gradients.push_back(atan2(grad_x.at<int>(Point(x,y)),grad_y.at<int>(Point(x,y)))*180/M_PI);
		}
	}

	int importantGradients = 0;
	double avgGradient = 0;
	for(int i=0; i<gradients.size(); i++)
	{
		if(gradients.at(i) != 0)
		{
			avgGradient += gradients.at(i);
			importantGradients++;
		}
	}
	avgGradient /= importantGradients;

	return avgGradient;
}

double gradientVariance(Mat &image)
{
	Mat src_gray = image.clone();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			if(getIntensity(getPixel(src_gray,x,y)) != 0)
				setPixel(src_gray,x,y,Vec3b(255,255,255));
		}
	}

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;

	/// Gradient X
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

	/// Gradient Y
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	vector<double> gradients;
	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			gradients.push_back(atan2(grad_x.at<int>(Point(x,y)),grad_y.at<int>(Point(x,y)))*180/M_PI);
		}
	}

	int importantGradients = 0;
	double avgGradient = 0;
	for(int i=0; i<gradients.size(); i++)
	{
		if(gradients.at(i) != 0)
		{
			avgGradient += gradients.at(i);
			importantGradients++;
		}
	}
	avgGradient /= importantGradients;

	
	double variance = 0;
	for(int i=0; i<gradients.size(); i++)
	{
		if(gradients.at(i) != 0)
			variance += pow(gradients.at(i)-avgGradient,2);
	}
	variance /= importantGradients;
	

	return variance;
}

double averageCurvature(Mat &image)
{
	Mat src_gray = image.clone();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			if(getIntensity(getPixel(src_gray,x,y)) != 0)
				setPixel(src_gray,x,y,Vec3b(255,255,255));
		}
	}

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat curv_x, curv_y;

	/// Gradient X
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( grad_x, curv_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

	/// Gradient Y
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	Sobel( grad_y, curv_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	double curvature = 0;
	double count = 0;
	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			double c = atan2(curv_x.at<int>(Point(x,y)),curv_y.at<int>(Point(x,y)))*180/M_PI;
			if(c != 0)
			{
				curvature += c;
				count++;
			}
		}
	}

	double avgCurvature = curvature/count;

	return avgCurvature;
}

double varianceCurvature(Mat &image)
{
	Mat src_gray = image.clone();

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			if(getIntensity(getPixel(src_gray,x,y)) != 0)
				setPixel(src_gray,x,y,Vec3b(255,255,255));
		}
	}

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat curv_x, curv_y;

	/// Gradient X
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( grad_x, curv_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

	/// Gradient Y
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	Sobel( grad_y, curv_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	double curvature = 0;
	double count = 0;
	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			double c = atan2(curv_x.at<int>(Point(x,y)),curv_y.at<int>(Point(x,y)))*180/M_PI;
			if(c != 0)
			{
				curvature += c;
				count++;
			}
		}
	}

	double avgCurvature = curvature/count;

	double variance = 0;
	for(int x=0; x<src_gray.cols; x++)
	{
		for(int y=0; y<src_gray.rows; y++)
		{
			double c = atan2(grad_x.at<int>(Point(x,y)),grad_y.at<int>(Point(x,y)))*180/M_PI;
			if(c != 0)
			{
				variance += pow(avgCurvature-c,2);
			}
		}
	}

	variance /= count;

	return variance;
}

double lineThickness(Mat &image)
{
	
	Mat img = image.clone();
	for(int x=0; x<img.cols; x++)
	{
		for(int y=0; y<img.rows; y++)
		{
			if(getIntensity(getPixel(img,x,y)) != 0)
				setPixel(img,x,y,Vec3b(255,255,255));
		}
	}
	cvtColor(img,img,CV_BGR2GRAY);
	
	Mat distTransform;
	Mat labels;
	distanceTransform(img,distTransform,labels,CV_DIST_L1,CV_DIST_MASK_PRECISE);

	double lineThickness = 0;
	double count = 0;
	for(int x=0; x<distTransform.cols; x++)
	{
		for(int y=0; y<distTransform.rows; y++)
		{
			if(distTransform.at<float>(Point(x,y)) != 0)
			{
				lineThickness += distTransform.at<float>(Point(x,y));
				count++;
			}
		}
	}

	double avgLineThickness = 0;
	if(count != 0)
	{
		avgLineThickness = lineThickness / count;
	}

	return avgLineThickness;
}

double lineThicknessVariance(Mat &image)
{
	
	Mat img = image.clone();
	for(int x=0; x<img.cols; x++)
	{
		for(int y=0; y<img.rows; y++)
		{
			if(getIntensity(getPixel(img,x,y)) != 0)
				setPixel(img,x,y,Vec3b(255,255,255));
		}
	}
	cvtColor(img,img,CV_BGR2GRAY);
	
	Mat distTransform;
	Mat labels;
	distanceTransform(img,distTransform,labels,CV_DIST_L1,CV_DIST_MASK_PRECISE);

	double lineThickness = 0;
	double count = 0;
	for(int x=0; x<distTransform.cols; x++)
	{
		for(int y=0; y<distTransform.rows; y++)
		{
			if(distTransform.at<float>(Point(x,y)) != 0)
			{
				lineThickness += distTransform.at<float>(Point(x,y));
				count++;
			}
		}
	}

	double avgLineThickness = 0;
	if(count != 0)
	{
		avgLineThickness = lineThickness / count;
	}
	
	double variance = 0;
	for(int x=0; x<distTransform.cols; x++)
	{
		for(int y=0; y<distTransform.rows; y++)
		{
			if(distTransform.at<float>(Point(x,y)) != 0)
			{
				variance += pow(avgLineThickness-double(distTransform.at<float>(Point(x,y))),2);
			}
		}
	}

	if(count != 0)
	{
		variance = variance / count;
	}

	return variance;
}

double letterSize(Mat &image)
{
	return double(double(image.rows)*double(image.cols));
}

//PLOT
void plotLines(string name, int width, int height, vector<Point> points)
{
	Mat graph = Mat::zeros(Size(width,height+20),CV_8UC3);
	int maxPos=0;
	int maxValue=0;
	for(int x=0; x<points.size(); x++)
	{
		if(points.at(x).y > maxValue)
		{
			maxPos = x;
			maxValue = points.at(x).y;
		}
	}
	for(int x=0; x<points.size(); x++)
	{
		points.at(x).y = height - double(points.at(x).y)/double(maxValue)*height + 10;
	}
	int x=0;
	while(x<points.size()-1)
	{
		line(graph,points.at(x),points.at(x+1),Scalar(255,255,255),1,8);
		x++;
	}
	imshow(name,graph);
}

void plotFeatureDistribution(string name, vector<Letter> &letters, int dim)
{
	Mat image = Mat::zeros(100,1000,CV_8UC3);
	Scalar blue = Scalar(255,0,0);
	Scalar red = Scalar(0,0,255);
	for(int i=0; i<letters.size(); i++)
	{
		if(letters.at(i).label == 1)
		{
			circle(image,Point(letters.at(i).featureVector.at(dim),40),1,Scalar(255,0,0));
		}
		if(letters.at(i).label == 2)
		{
			circle(image,Point(letters.at(i).featureVector.at(dim),60),1,Scalar(0,0,255));
		}
	}
	imshow(name,image);
}

void sortDistancesAndLabels(vector<double> &distances, vector<int> &labels, int p, int q)
{
	int r;
    if(p<q)
    {
        r=partition(distances,labels,p,q);
        sortDistancesAndLabels(distances,labels,p,r);  
        sortDistancesAndLabels(distances,labels,r+1,q);
    }
}

int partition(vector<double> &distances, vector<int> &labels, int p, int q)
{
	int x= distances[p];
    int i=p;
    int j;

    for(j=p+1; j<q; j++)
    {
        if(distances[j]<=x)
        {
            i=i+1;
            swap(distances[i],distances[j]);
			swap(labels[i],labels[j]);
        }

    }
    swap(distances[i],distances[p]);
	swap(labels[i],labels[p]);
    return i;
}

double getMinFeatureValue(vector<double> &featureDimension)
{
	int minValue = featureDimension.at(0);
	for(int i=0; i<featureDimension.size(); i++)
	{
		if(minValue > featureDimension.at(i))
		{
			minValue = featureDimension.at(i);
		}
	}
	return minValue;
}

double getMaxFeatureValue(vector<double> &featureDimension)
{
	int maxValue = featureDimension.at(0);
	for(int i=0; i<featureDimension.size(); i++)
	{
		if(maxValue < featureDimension.at(i))
		{
			maxValue = featureDimension.at(i);
		}
	}
	return maxValue;
}

void normalizeDimension(vector<Letter> &letters, int dim)
{
	vector<double> featureDimension;
	for(int i=0; i<letters.size(); i++)
	{
		featureDimension.push_back(letters.at(i).featureVector.at(dim));
	}

	/*
	double dimMin = getMinFeatureValue(featureDimension);
	double dimMax = getMaxFeatureValue(featureDimension);
	for(int i=0; i<featureDimension.size(); i++)
	{
		featureDimension.at(i) -= dimMin;
		featureDimension.at(i) = featureDimension.at(i) / (dimMax-dimMin) * 1000;
	}
	*/

	normalize(featureDimension,featureDimension,0,1000,NORM_MINMAX);

	for(int i=0; i<letters.size(); i++)
	{
		letters.at(i).featureVector.at(dim) = featureDimension.at(i);
	}
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


double knnClassifyValidation(vector<Letter> letters, int k, double decisionTolerance, bool isEqual, bool errorOnLetters)
{
	double error = 0;
	int matchingLetters = 0;

	for(int pos=0; pos<letters.size(); pos++)
	{

		vector<vector<double>> featureVectors;
		vector<int> labels;
		for(int i = 0; i<letters.size(); i++)
		{
			featureVectors.push_back(letters.at(i).featureVector);
			labels.push_back(letters.at(i).label);
		}

		vector<double> distances;
		for(int i=0; i<featureVectors.size(); i++)
		{
			double distance = 0;
			for(int j=0; j<featureVectors.at(i).size(); j++)
			{
				distance += pow(featureVectors.at(i).at(j) - letters.at(pos).featureVector.at(j),2);
			}
			distance = sqrt(distance);
			distances.push_back(distance);
		}
		sortDistancesAndLabels(distances,labels,0,distances.size());

		vector<int> predictionSet;
		for(int i=1; i<k+1; i++)
		{
			predictionSet.push_back(labels.at(i));
		}

		int count_c1 = 0;
		int count_c2 = 0;
		for(int i=0; i<predictionSet.size(); i++)
		{
			if(predictionSet.at(i) == 1)
				count_c1++;
			if(predictionSet.at(i) == 2)
				count_c2++;
		}

		int maxCount = 0;
		if(count_c1 > count_c2)
		{
			maxCount = count_c1;
		}
		if(count_c2 > count_c1)
		{
			maxCount = count_c2;
		}
		if(count_c1 == count_c2)
		{
			maxCount = count_c1;
		}

		double confidence = double(double(maxCount)/double(predictionSet.size()));
		if(confidence >= 0.5-decisionTolerance && confidence <= 0.5+decisionTolerance)
		{
			matchingLetters++;
		}
	}

	if(errorOnLetters)
	{
		if(isEqual)
		{
			error = letters.size() - matchingLetters;
		}
		else
		{
			error = matchingLetters;
		}
		error /= double(letters.size());
	}
	else
	{
		double confidence = double(double(matchingLetters)/double(letters.size()));
		if(!isEqual && confidence>0.5)
		{
			error = 1;
		}
		if(isEqual && confidence<0.5)
		{
			error = 1;
		}
	}

	return error;
}

Manuscript getManuscript(string fileName, int author)
{	
	Manuscript manuscript;
	manuscript.fileName = fileName;
	manuscript.author = author;
	manuscript.image = imread(manuscript.fileName);
	return manuscript;
}

struct TrainingError{
	double error;
	int k;
	double sigma;
};

void minimizeTrainingError(int kMin, int kMax, int kStep, double sigmaMin, double sigmaMax, double sigmaStep, bool errorOnLetters)
{
	vector<Manuscript> manuscripts;
	manuscripts.push_back(getManuscript("front/0.jpg",1));
	manuscripts.push_back(getManuscript("front/1.jpg",1));
	manuscripts.push_back(getManuscript("front/2.jpg",1));
	manuscripts.push_back(getManuscript("front/3.jpg",1));
	manuscripts.push_back(getManuscript("front/4.jpg",1));
	manuscripts.push_back(getManuscript("front/5.jpg",1));

	manuscripts.push_back(getManuscript("back/0.jpg",2));
	manuscripts.push_back(getManuscript("back/1.jpg",2));
	manuscripts.push_back(getManuscript("back/2.jpg",2));
	manuscripts.push_back(getManuscript("back/3.jpg",2));
	manuscripts.push_back(getManuscript("back/4.jpg",2));
	manuscripts.push_back(getManuscript("back/5.jpg",2));

	for(int i=0; i<manuscripts.size(); i++)
	{
		detectChineseLetters(manuscripts.at(i),0,0);
		for(int j=0; j<manuscripts.at(i).letters.size(); j++)
		{
			manuscripts.at(i).letters.at(j).featureVector = computeFeatureVector(manuscripts.at(i).letters.at(j).image);
		}
	}

	vector<TrainingError> trainingErrors;
	bool visualization = 0;

	cout << "========================================" << endl;
	cout << "Training-Error-Minimization..." << endl;
	cout << "========================================" << endl;
	cout << "kMin: " << kMin << endl;
	cout << "kMax: " << kMax << endl;
	cout << "kStep: " << kStep << endl;
	cout << "sigmaMin: " << sigmaMin << endl;
	cout << "sigmaMax: " << sigmaMax << endl;
	cout << "sigmaStep: " << sigmaStep << endl;
	cout << "onLetters: " << errorOnLetters << endl;
	cout << "========================================" << endl;

	for(double sigma=sigmaMin; sigma <= sigmaMax; sigma += sigmaStep)
	{
		for(int k=kMin; k<=kMax; k += kStep)
		{
			TrainingError trainingError;
			trainingError.error = 0.0;
			trainingError.k = k;
			trainingError.sigma = sigma;

			double runs = 0;
			cout << "Iteration: k = " << k << " and sigma = " << sigma << "..." << endl;
			for(int i = 0; i<manuscripts.size(); i++)
			{
				for(int j = i; j<manuscripts.size(); j++)
				{
					//cout << "Comparing " << manuscripts.at(i).fileName << " with " << manuscripts.at(j).fileName << "..." << endl;

					//Gather all letters
					vector<Letter> letters;
					for(int pos=0; pos<manuscripts.at(i).letters.size(); pos++)
					{
						letters.push_back(manuscripts.at(i).letters.at(pos));
						letters.back().label = 1;
					}
					for(int pos=0; pos<manuscripts.at(j).letters.size(); pos++)
					{
						letters.push_back(manuscripts.at(j).letters.at(pos));
						letters.back().label = 2;
					}

					//Normalize Feature Vectors Dimensions
					normalizeDimension(letters,0);
					normalizeDimension(letters,1);
					normalizeDimension(letters,2);
					normalizeDimension(letters,3);
					normalizeDimension(letters,4);
					normalizeDimension(letters,5);
					normalizeDimension(letters,6);
					normalizeDimension(letters,7);
					normalizeDimension(letters,8);
					normalizeDimension(letters,9);
					normalizeDimension(letters,10);
					normalizeDimension(letters,11);
					normalizeDimension(letters,12);

					//kNN Classification
					bool areEqual = manuscripts.at(i).author == manuscripts.at(j).author;
					trainingError.error += knnClassifyValidation(letters,k,sigma,areEqual, errorOnLetters);
					//cout << "False Classified: " << falseClassified << " of " << letters.size() << " (Error: " << error << ")" << endl;

					runs++;
				}
			}
			trainingError.error /= runs;
			trainingErrors.push_back(trainingError);

			cout << "Training Error: " << trainingError.error << endl;
			cout << "----------------------------------------" << endl;
		}
	}

	int posMin = 0;
	double minError = trainingErrors.at(0).error;
	for(int i=1; i<trainingErrors.size(); i++)
	{
		if(trainingErrors.at(i).error < minError)
		{
			posMin = i;
			minError = trainingErrors.at(i).error;
		}
	}

	cout << "Minimal Training Error for k = " << trainingErrors.at(posMin).k << " and sigma = " << trainingErrors.at(posMin).sigma << " with E=" << minError << endl;
	cout << "========================================" << endl;
}