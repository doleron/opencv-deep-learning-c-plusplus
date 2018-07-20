#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace dnn;

void loadClassNames(string classNamesFile, vector<string>& classesVector);

int main(int argc, char** argv) {

	string inputImageFile = "images/pelicans.jpg";

	//in this example we'll use the MobileNet on caffe
	String model = "MobileNetSSD_deploy.caffemodel";
	String config = "MobileNetSSD_deploy.prototxt.txt";

	//The settings for MobileNet can be found here: https://github.com/opencv/opencv/tree/master/samples/dnn
	float scale = 0.007843;
	Scalar mean = Scalar(127.5, 127.5, 127.5);
	Size size = Size(300, 300);

	vector<string> classes;
	string classesFile = "mobilenet_classes.txt";
	loadClassNames(classesFile, classes);

	Net net = readNet(model, config);

	Mat image = imread(inputImageFile);

	if (image.empty()) {
		CV_Error(Error::StsError, "Cannot load image from " + inputImageFile);
	}

	Mat blob;
	blobFromImage(image, blob, scale, size, mean, false, false);

	net.setInput(blob);
	Mat prob = net.forward("detection_out");

	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
			detection.ptr<float>());

	Mat output;
	image.copyTo(output);
	float confidenceThreshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold) {

			int idx = static_cast<int>(detectionMat.at<float>(i, 1));
			string clazz = classes[idx];
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3)
					* image.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4)
					* image.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5)
					* image.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6)
					* image.rows);

			Rect object((int) xLeftBottom, (int) yLeftBottom,
					(int) (xRightTop - xLeftBottom),
					(int) (yRightTop - yLeftBottom));

			rectangle(output, object, Scalar(0, 255, 0), 2);

			string label = format("%s: %.2f %%", clazz.c_str(), confidence * 100);
			putText(output, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		}
	}

	imshow("Result", output);
	waitKey();

	return 0;
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
