#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

//Required function declarations
void detectAndDisplay(Mat, int);	//Main detect function which applies Haar-Cascade
void verifySignals(Mat&, vector<Rect>&, vector<Rect>&);	//For verifying output of cascade is an actual traffic light
void findLightColor(Mat&, vector<Rect>&, int);	//Finds the status of traffic light after verification function
void showOutput();	
void validateResults();		//Validator function for computing various metrics...

vector<Point> circleContour;
String cascade_name;
CascadeClassifier lights_cascade;
String window_name = "Image - Light detection";
int randomIterator = 0;

struct signalStats
{
	int imageNumber = 0;
	Rect dimensions;
	bool isGreen = false;
	bool isRed = false;
	bool isAmber = false;
};

vector<signalStats> detectedLights, realLights;
vector<vector<signalStats>> realDataSplit, detectedDataSplit;
int detectedRectsNum = 0;
int correctColor = 0;
string saveLoc = "output/";

int main(int argc, const char** argv)
{
	vector<Mat> images;
	for (int i = 0; i < 14; i++) {
		string loc = "images/";
		string ext = ".png";
		int numImage = i + 1;
		string path = loc.append(to_string(numImage));
		path.append(ext);
		Mat temp;
		temp = imread(path, IMREAD_UNCHANGED); // Read the file
		if (temp.empty()) // Check for invalid input
		{
			cout << "Could not open or find the image" << endl;
			continue;
		}
		images.push_back(temp);
	}

	Mat circleImage = imread("circle.png", IMREAD_GRAYSCALE);
	vector<Vec4i> hierarchy_c;
	vector<vector<Point>> circleCont;
	findContours(circleImage, circleCont, hierarchy_c, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	circleContour = circleCont[0];

	cascade_name = "haar4_20.xml";
	if (!lights_cascade.load(cascade_name)) { printf("--(!)Error loading cascade\n"); return -1; };
	
	for(int i = 0; i < images.size(); i++)
		detectAndDisplay(images[i], i);
	showOutput();
	validateResults();

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

void showOutput() {
	cout << "\n-------------------------------------------------------\n";
	cout << "\tDETECTED LIGHTS - " << detectedLights.size() << endl;
	cout << "-------------------------------------------------------\n";
	string previousDetected = "";

	for (int i = 0; i < detectedLights.size(); i++) {
		cout << " IMAGE NUMBER " << detectedLights[i].imageNumber << endl;
		cout << "\tLIGHT NUMBER - " << (i + 1) << endl;
		string colorStatus;
		if (detectedLights[i].isRed && !detectedLights[i].isAmber) {
			colorStatus = "Red";
			previousDetected = colorStatus;
		}
		else if (detectedLights[i].isGreen) {
			colorStatus = "Green";
			previousDetected = colorStatus;
		}
		else if (detectedLights[i].isAmber && !detectedLights[i].isRed) {
			colorStatus = "Amber";
			previousDetected = colorStatus;
		}
		else {
			colorStatus = "Red+Amber";
			previousDetected = colorStatus;
		}

		//Might misclassify the color in very rare cases...
		if (detectedLights[i].isRed == false && detectedLights[i].isAmber == false && detectedLights[i].isGreen == false)
			if (!previousDetected.compare("")==0)
				colorStatus = previousDetected;
		cout << "\t\tCOLOR STATUS IS " << colorStatus << endl;
	}
}

void validateResults() {
	//cout << "\n============================================";
	//cout << "\n\nREADING GROUND TRUTH...\n";
	ifstream myfile("images/truth.csv");
	if (!myfile.is_open()) {
		cout << "Cannot open CSV file";
		return;
	}
	string token;
	
	//SKIP FIRST 3 LINES
	getline(myfile, token);
	getline(myfile, token);
	getline(myfile, token);

	//READ DATA
	int prevImageNumber = 0;
	while (!myfile.eof()) {
		signalStats temp;

		//Image Number
		getline(myfile, token, ',');
		int imageNumber = 0;
		if (token.compare("") == 0)
			imageNumber = prevImageNumber;
		else
			imageNumber = stoi(token);
		prevImageNumber = imageNumber;
		temp.imageNumber = imageNumber;
		//cout << " IMAGE NUMBER " << imageNumber << endl;

		//State
		getline(myfile, token, ',');
		if (token.compare("Green") == 0 || token.compare("green") == 0)
			temp.isGreen = true;
		if (token.compare("Red") == 0 || token.compare("red") == 0)
			temp.isRed = true;
		if (token.compare("Amber") == 0 || token.compare("amber") == 0)
			temp.isAmber = true;
		if (token.compare("Red+Amber") == 0 || token.compare("red+amber") == 0) {
			temp.isAmber = true;
			temp.isRed = true;
		}
		

		//Get enclosing rectangles...
		//Ignoring previous 4 because considering only FULL LIGHT conditions...
		int topLeftC, topLeftR, bottomRightC, bottomRightR = 0;
		getline(myfile, token, ',');
		topLeftC = stoi(token);
		getline(myfile, token, ',');
		topLeftR = stoi(token);
		getline(myfile, token, ',');
		bottomRightC = stoi(token);
		getline(myfile, token, ',');
		bottomRightR = stoi(token);

		int rectWidth = bottomRightC - topLeftC + 1;
		int rectHeight = bottomRightR - topLeftR + 1;


		temp.dimensions.x = topLeftC;
		temp.dimensions.y = topLeftR;
		temp.dimensions.width = rectWidth;
		temp.dimensions.height = rectHeight;

		realLights.push_back(temp);

		getline(myfile, token, ',');
		getline(myfile, token, ',');
		getline(myfile, token, ',');
		getline(myfile, token);
	}

	int currentNumber = 0;
	//SPLITTING DETECTED LIGHT DATA INTO RESPECTIVE IMG NUMBERS FOR EASY VALIDATION
	currentNumber = realLights[0].imageNumber;
	realDataSplit.push_back(vector<signalStats>());
	realDataSplit[0].push_back(realLights[0]);
	for (int i = 1; i < realLights.size(); i++) {
		if (realLights[i].imageNumber == currentNumber) {
			realDataSplit[currentNumber - 1].push_back(realLights[i]);
		}
		else {
			realDataSplit.push_back(vector<signalStats>());
			currentNumber = realLights[i].imageNumber;
			realDataSplit[currentNumber - 1].push_back(realLights[i]);
		}
	}

	currentNumber = 0;
	//SPLITTING DETECTED LIGHT DATA INTO RESPECTIVE IMG NUMBERS FOR EASY VALIDATION
	currentNumber = detectedLights[0].imageNumber;
	detectedDataSplit.push_back(vector<signalStats>());
	detectedDataSplit[0].push_back(detectedLights[0]);
	for (int i = 1; i < detectedLights.size(); i++) {
		if (detectedLights[i].imageNumber == currentNumber) {
			detectedDataSplit[currentNumber - 1].push_back(detectedLights[i]);
		}
		else {
			for(int temp = 0; temp < (detectedLights[i].imageNumber - currentNumber); temp++)
				detectedDataSplit.push_back(vector<signalStats>());
			currentNumber = detectedLights[i].imageNumber;
			detectedDataSplit[currentNumber - 1].push_back(detectedLights[i]);
		}
	}

	//Now that data is split nicely, we can validate our metrics...
	//First let's calculate number of signals found...this will be the signal detection accuracy
	for (int i = 0; i < detectedDataSplit.size(); i++) {
		//First obtain real signals corresponding to the image...
		vector<signalStats> realSignals;
		for (int temp = 0; temp < realDataSplit[i].size(); temp++) {
			realSignals.push_back(realDataSplit[i][temp]);
		}
		//To check False Positives
		vector<bool> isPresent;
		if (detectedDataSplit[i].size() > 0) {
			for (int j = 0; j < detectedDataSplit[i].size(); j++) {
				for (int temp = 0; temp < realSignals.size(); temp++) {
					Rect overlap = detectedDataSplit[i][j].dimensions & realSignals[temp].dimensions;
					double realArea = realSignals[temp].dimensions.area();
					double overlapArea = overlap.area();
					//cout << "\n CHECKING OVERLAP FOR REAL "<< temp+1<<" DATA - " << overlap.x << " " << overlap.y 
					//	<< "\n OVERLAP AREA " << overlap.area() 
					//	<< "\n REAL RECT AREA " << realRects[temp].area() << endl;
					double overlapval = overlapArea/realArea;
					//cout << "  OVERLAP VALUE = " << overlapval << endl;
					if (overlapval > 0.8) {
						if(realSignals[temp].isAmber == detectedDataSplit[i][j].isAmber &&
							realSignals[temp].isRed == detectedDataSplit[i][j].isRed &&
							realSignals[temp].isGreen == detectedDataSplit[i][j].isGreen)
							correctColor++;
						detectedRectsNum++;
					}
				}
			}
		}
	}
	int totalDetected = detectedLights.size();
	int falsePositives = totalDetected - detectedRectsNum;
	double fpr = (double)falsePositives / (double)totalDetected;	//Proportion detected falsely as a traffic light...
	double colorAcc = (double)correctColor / (double)detectedRectsNum;	//Check color acc. of detected signals...
	double colorAccTotal = (double)correctColor / (double)realLights.size();	//Check color acc. overall (not detected are 0)
	double diceCoeff = (double)(2 * detectedRectsNum) / (double)(totalDetected + realLights.size());

	cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\t\tDATA"
		<< " METRICS\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

	cout << "\n   ACCURACY (SIGNAL DETECTION) = " 
		<< (double) ( (double)detectedRectsNum / (double) realLights.size()) * 100.0 << "%" << endl;

	cout << "\n =============================================\n\n   FALSE POSITIVES RATE (SIGNAL DETECTION) = " 
		<< (double) fpr * 100.0 << "%" << endl;

	cout << "\n =============================================\n\n   COLOR ACCURACY (IN DETECTED SIGNALS) = " 
		<< colorAcc * 100 << "%" << endl;

	cout << "\n =============================================\n\n   COLOR ACCURACY (OVERALL) = "
		<< colorAccTotal * 100 << "%" << endl;

	cout << "\n =============================================\n\n   DICE COEFFICIENT = "
		<< diceCoeff << "%" << endl;

	//double colorAcc = getColorAccuracy();
}

void detectAndDisplay(Mat frame, int imageNumber)
{
	vector<Rect> signal;
	vector<Rect> finalSignals;
	Mat frame_real;
	//Storing the original image before manipulation
	frame.copyTo(frame_real);

	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	vector<int> reject_levels;
	vector<double> weights;
	vector<Mat> signalImgs;
	lights_cascade.detectMultiScale(frame, signal, reject_levels, weights, 1.05, 6, 0 | CASCADE_SCALE_IMAGE, Size(24, 64), Size(65,180), true);
	for (size_t i = 0; i < signal.size(); i++)
	{
		//Discarding found rectangles with low weights...
		if (weights[i] > 1) {
			//rectangle(frame, signal[i], Scalar(0, 0, 255), 2, 8, 0);
			finalSignals.push_back(signal[i]);
			signalImgs.push_back(frame_real(signal[i]));
		}
		//cout << " RECT "<< i+1 <<" REJECT LEVEL - " << reject_levels[i] << ", WEIGHT - " << weights[i] << endl;
	}
	//Check if any signals were detected. If none, run detectMultiScale again...
	if (finalSignals.size() == 0) {
		lights_cascade.detectMultiScale(frame, signal, reject_levels, weights, 1.05, 3, 0 | CASCADE_SCALE_IMAGE, Size(24, 64), Size(65, 180), true);
		for (size_t i = 0; i < signal.size(); i++)
		{
			if (weights[i] > 1) {
				finalSignals.push_back(signal[i]);
				signalImgs.push_back(frame_real(signal[i]));
			}
		}
	}

	//Returns vector of rectangles which are traffic lights
	vector<Rect> realLights;
	verifySignals(frame_real, finalSignals, realLights);
	findLightColor(frame_real, realLights, imageNumber);
}

void verifySignals(Mat& frame, vector<Rect>& rects, vector<Rect>& output) {
	//This vector will contain detected traffic signals and return them...
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Vec3f> circles;
	Mat frameGray;
	cvtColor(frame, frameGray, CV_BGR2GRAY);
	//cout << "\n" << rects.size();
	for (size_t i = 0; i < rects.size(); i++) {
		bool rectHasLight = false;
		Mat temp = frameGray(rects[i]);
		Mat fr;
		GaussianBlur(temp, fr, Size(3, 3), 0, 0);
		Mat thresh;
		//adaptiveThreshold(fr, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 2);
		threshold(fr, thresh, 80, 255, THRESH_BINARY);
		//imshow(to_string(i + 1), thresh);
		findContours(thresh, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		//cout << "IMG " << i + 1 << " CONTOUR SIZE - " << contours.size() << endl;
		if(contours.size()>0 && contours.size() <= 10)	//JUST ESTIMATING THAT A TRAFFIC SIGNAL WILL CONTAIN LESS THAN 10 CONTOURS...
			for (int j = 0; j < contours.size(); j++) {
				if (contourArea(contours[j]) > 10) {	//15 BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!
					if (matchShapes(contours[j], circleContour, CV_CONTOURS_MATCH_I1, 0) < 0.1)
						rectHasLight = true;
				}
			}
		if (rectHasLight) {
			output.push_back(rects[i]);
			Scalar color(0, 0, 255);
			rectangle(frame, rects[i], color, 2, 8, 0);

		}
		contours.clear();
		hierarchy.clear();
	}
	randomIterator++;
	string writeLoc = saveLoc;
	writeLoc.append(to_string(randomIterator));
	writeLoc.append(".png");
	imwrite(writeLoc, frame);
}

void findLightColor(Mat& real_frame, vector<Rect>& signals, int imageNumber) {
	Mat frame;
	real_frame.copyTo(frame);
	vector<Mat> lightsBGR;

	//OBTAIN SIGNAL IMAGES IN HSV
	//cout << "\nINSIDE FIND COLOR... NUMBER OF SIGNALS - " << signals.size() <<endl;

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(5, 5));

	for (int i = 0; i < signals.size(); i++) {
		Mat temp;
		cvtColor(frame(signals[i]), temp, COLOR_BGR2HSV);
		morphologyEx(temp, temp, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(2,2), Point(-1,-1)));
		morphologyEx(temp, temp, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1, -1)));

		lightsBGR.push_back(temp);
		//imshow(to_string(i + 10), temp);
	}

	for (int i = 0; i < lightsBGR.size(); i++) {
		signalStats temp;
		Mat lower_green_range, upper_green_range, lower_red_range, upper_red_range, lower_yellow_range, upper_yellow_range;
		Mat red_range, green_range;
		int sensitivity = 25;
		bool greenMatch = false;
		int redIndex, yellowIndex, greenIndex = 0;

		//FOR YELLOW
		inRange(lightsBGR[i], Scalar(45 - sensitivity, 80, 80), Scalar(45 + sensitivity, 255, 255), lower_yellow_range);
		//FOR RED
		inRange(lightsBGR[i], Scalar(0, 100, 100), Scalar(20, 255, 255), lower_red_range);
		inRange(lightsBGR[i], Scalar(150, 100, 100), Scalar(180, 255, 255), upper_red_range);
		//Using two inRange images for RED because red wraps around the HSV colorspace
		addWeighted(lower_red_range, 1.0, upper_red_range, 1.0, 0.0, red_range);

		string tagY = "YELLOW ";
		tagY.append(to_string(i + 1));
		string tagR = "RED ";
		tagR.append(to_string(i + 1));
		string tagG = "GREEN ";
		tagG.append(to_string(i + 1));

		vector<vector<Point>> contoursY;
		vector<Vec4i> hierarchyY;
		morphologyEx(lower_yellow_range, lower_yellow_range, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		morphologyEx(lower_yellow_range, lower_yellow_range, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		findContours(lower_yellow_range, contoursY, hierarchyY,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		if (contoursY.size() > 0 && contoursY.size() <= 10)	//JUST ESTIMATING THAT A TRAFFIC SIGNAL WILL CONTAIN LESS THAN 10 CONTOURS...
			for (int j = 0; j < contoursY.size(); j++) {
				Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
				if (matchShapes(contoursY[j], circleContour, CV_CONTOURS_MATCH_I1, 0) < 0.1)
					if (contourArea(contoursY[j]) > 15) {
						temp.isAmber = true;
						yellowIndex = j;
					}
			}
		//imshow(tagY, lower_yellow_range);

		vector<vector<Point>> contoursR;
		vector<Vec4i> hierarchyR;
		//GaussianBlur(red_range, red_range, Size(5, 5), 0, 0);
		morphologyEx(red_range, red_range, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		morphologyEx(red_range, red_range, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		findContours(red_range, contoursR, hierarchyR, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		if (contoursR.size() > 0 && contoursR.size() <= 10)	
			for (int j = 0; j < contoursR.size(); j++) {
				Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
				if (matchShapes(contoursR[j], circleContour, CV_CONTOURS_MATCH_I1, 0) < 0.1)
					if (contourArea(contoursR[j]) > 15) {
						temp.isRed = true;
						redIndex = j;
					}
			}
		//imshow(tagR, red_range);

		//FOR GREEN
		inRange(lightsBGR[i], Scalar(60 - sensitivity, 60, 80), Scalar(100, 255, 255), lower_green_range);
		inRange(lightsBGR[i], Scalar(0, 255, 0), Scalar(230, 255, 205), upper_green_range);
		addWeighted(lower_green_range, 1.0, upper_green_range, 1.0, 0.0, green_range);
		vector<vector<Point>> contoursG;
		vector<Vec4i> hierarchyG;
		//GaussianBlur(lower_green_range, lower_green_range, Size(5, 5), 0, 0);
		morphologyEx(green_range, green_range, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		morphologyEx(green_range, green_range, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1)));
		findContours(green_range, contoursG, hierarchyG, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		//Mat imgContours = Mat::zeros(lower_green_range.size(), CV_8UC3);
		if (contoursG.size() > 0 && contoursG.size() <= 5)
			for (int j = 0; j < contoursG.size(); j++) {
				Scalar color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
				//drawContours(imgContours, contoursG, j, color, 1, 8, hierarchyG);
				if (matchShapes(contoursG[j], circleContour, CV_CONTOURS_MATCH_I1, 0) < 0.1)
					if (contourArea(contoursG[j]) > 15) {
						temp.isGreen = true;
						greenMatch = true;
						greenIndex = j;
					}
			}
		//Reverification for green because of some really bright images...
		if (!greenMatch) {
			int greenpixels = countNonZero(green_range);
			if (greenpixels > 10)
				temp.isGreen = true;
		}
		//imshow(tagG, lower_green_range);

		//Some reverification between red and amber...
		if (temp.isAmber && temp.isRed) {
			Mat debug;
			lightsBGR[i].copyTo(debug);
			Rect redBox = boundingRect(contoursR[redIndex]);
			Rect yellowBox = boundingRect(contoursY[yellowIndex]);
			Rect overlap = redBox & yellowBox;
			double overlapRatio1 = ((double) overlap.area()) / ((double)redBox.area());
			double overlapRatio2 = ((double)overlap.area()) / ((double) yellowBox.area());
			double ratio;
			overlapRatio1 > overlapRatio2 ? ratio = overlapRatio1 : ratio = overlapRatio2;
			//cout << " OVERLAP RATIOSS IN " << imageNumber << " ARE " << overlapRatio1 << " AND " << overlapRatio2 << endl;
			if (ratio > 0.9) {	//Light is either RED or YELLOW but NOT BOTH.
				if (contourArea(contoursR[redIndex]) > contourArea(contoursY[yellowIndex]))
					temp.isAmber = false;
				else
					temp.isRed = false;
			}
			//imshow("DEBUG", debug);
		}

		//Extra verification for RED 
		if (temp.isRed && !temp.isAmber) {
			int top = (lightsBGR[i].rows / 3) + 5;
			int bottom = (top - 5)*2 - 5;
			//cout << " TOP: " << top << " BOTTOM: " << bottom << endl;
			Rect redBox = boundingRect(contoursR[redIndex]);
			int yCenter = redBox.y + (redBox.height / 2);
			//cout << "Y CENTER: " << yCenter << endl;
			if (yCenter > top && yCenter < bottom) {
				temp.isRed = false;
				temp.isAmber = true;
			}
		}

		temp.imageNumber = imageNumber + 1;
		temp.dimensions = signals[i];	//LIGHTSBGR and SIGNALS have SAME SIZE so its k to do this..I guess
		detectedLights.push_back(temp);
	}


}