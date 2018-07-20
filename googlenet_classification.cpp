#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;

using namespace cv;
using namespace dnn;

//prototype of function to return the index of the class with max confidence
void classIndexWithHigherConfidence(Mat & prob, int *classId, double *confidence);

//convenience method to load class names
void loadClassNames(string classNamesFile, vector<string>& classesVector);

int main(int argc, char** argv) {

	//the image which we want to classify
	string inputImageFile = "images/fighters.jpg";

	//in this example we'll use the googlenet on the caffe framework
    String model = "bvlc_googlenet.caffemodel";
    String config = "bvlc_googlenet.prototxt";

	//specific parameters of the network in use (googlenet on caffemodel)
	//this settings can be found in the training settings file
	//see here https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt
    float scale = 1;
    Scalar mean = Scalar(104, 117, 123);
    Size size = Size(244, 244);

    //a vector to store the classe names
    vector<string> classes;
    //the file with the list of class names
	string classesFile = "synset_words.txt";
	loadClassNames(classesFile, classes);

	//finally load the network
    Net net = readNet(model, config);

    //now let's load the query image
    Mat image = imread(inputImageFile);

	if (image.empty()) {
		CV_Error(Error::StsError, "Cannot load image from " + inputImageFile);
	}

	//In this step the query image is changed to fit to the network expected input format
	//the cv::dnn::blobFromImage is a convenience function to perform it easier
	Mat blob;
	blobFromImage(image, blob, scale, size, mean, false, false);

	//passing the transformed query image to the network and calling forward to
	//perform the propagation step trhough the network and store the result of layer 'prob'
	//in the resulting Mat
	net.setInput(blob);
	Mat prob = net.forward("prob");

	//the result prob Mat has probabilities for each class
	//So, it is required to find the more probable class (i.e, the class with max confidence)
	double confidence;
	int classId;
	classIndexWithHigherConfidence(prob, &classId, &confidence);
	string clazz = classes[classId];

	//By default the network store metrics of execution performance time in terms of CPU cycles
	//here
	vector<double> spentTimePerLayer;
	long spentCycles = net.getPerfProfile(spentTimePerLayer);

	//convert cycle time in millisecond
	double cpuFrequence = getTickFrequency() / 1000.0;
	double spentTime = spentCycles / cpuFrequence;

	//now the remain work is just to output the results is a friendly way
	string confidenceMessage = format("Confidence %.2f %%", confidence * 100);
	string performanceMessage = format("Took: %.2f ms", spentTime);
	cout << clazz << endl;
	cout << confidenceMessage << endl;
	cout << performanceMessage << endl;

	int outputHeight = image.size().height + 50;
	int outputWidth = image.size().width;
	Size outputSize(outputWidth, outputHeight);
	Mat output(outputSize, image.type());
	image.copyTo(output(Rect(0, 50, image.cols, image.rows)));

	rectangle(output, Point(0, 0), Point(output.size().width, 50), Scalar(0, 255, 255), -1);

	putText(output, clazz, Point(3, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
	putText(output, confidenceMessage, Point(3, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
	putText(output, performanceMessage, Point(3, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

	imshow("Result", output);
	waitKey();

	return 0;
}

void classIndexWithHigherConfidence(Mat & prob, int *classId, double *confidence) {
	Point classIdPoint;
	//reshape the blob to 1x1000 matrix
	//here this step is just ilustrative since that googlenet probs layer
	//is 1x1000 already
	Mat flatProbMat = prob.reshape(1, 1);
	minMaxLoc(flatProbMat, 0, confidence, 0, &classIdPoint);
	*classId = classIdPoint.x;
}

void loadClassNames(string classNamesFile, vector<string>& classesVector) {

	ifstream ifs(classNamesFile.c_str());
	if (!ifs.is_open()) {
		CV_Error(Error::StsError, "File " + classNamesFile + " not found");
	}
	string line;
	while (getline(ifs, line)) {
		classesVector.push_back(line);
	}
}
