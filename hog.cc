#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace boost;

std::vector<cv::Mat> LoadSampleImages(const std::string& directory) {
	filesystem::path dir_path{directory};
	std::vector<cv::Mat> samples;
	if (filesystem::is_directory(dir_path)) {
		for (auto&& path : filesystem::directory_iterator(dir_path)) {
			if (filesystem::is_regular_file(path)) {
				auto filename = path.path().relative_path().string();
				auto m = cv::imread(filename, 0);
				if (m.empty()) continue;
				samples.push_back(std::move(m));
			}
		}
	}
	return samples;
}

void GetSVMDetector(const Ptr<SVM>& svm, vector<float>& hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

Mat ComputeHog(const std::vector<cv::Mat> & samples, cv::Size size) {
	Mat train_data;
	for (auto img_raw : samples) {
		cv::resize(img_raw, img_raw, size);
		cv::HOGDescriptor d;
		d.winSize = size;
		std::vector<float> despv;
		std::vector<cv::Point> locations;
		try {
		d.compute(img_raw, despv, cv::Size(8,8), cv::Size(0,0), locations);
		}
		catch(std::exception& e) {
			std::cerr << e.what() << endl;
			return train_data;
		}
		auto row = Mat{1,static_cast<int>(despv.size()),CV_32FC1,despv.data()};
		train_data.push_back(row);
	}
	return train_data;
}

void train_svm(const Mat& train_data,
		const std::vector<int>& labels,
		const std::string& modelfile) {
	/* Default values to train SVM */
	cv::Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::ONE_CLASS); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train_data, ROW_SAMPLE, cv::Mat{labels});
	svm->save(modelfile);
}

void VehicleDetect(const Size& size,
		const std::string& videoname,
		const std::string& modelname) {
		
	auto hog = HOGDescriptor{};
	hog.winSize = size;
	auto svm = StatModel::load<SVM>(modelname);
	auto video = VideoCapture{videoname};
	auto hog_detector = vector<float>{};
	GetSVMDetector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);

	while (true) {
		auto img = Mat{};
		video.read(img);
		if (img.empty()) break;
		auto draw = Mat{img};
		auto locations = vector<Rect>{};
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		hog.detectMultiScale(img, locations);
		if (!locations.empty()) {
			for (const auto& box : locations) {
				rectangle(draw, box, Scalar(0,0,255),2);
			}
		}
		imshow(__func__,draw);
		auto key = waitKey(1);
		if (key == 'k') break;
	}
}

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    cout << "Usage: test path\n";
    return 1;
  }
	//auto size = cv::Size(24,16);
	auto size = cv::Size(32,32);
  auto samples = LoadSampleImages(argv[1]);
	std::vector<int> labels(samples.size(), 1);
	auto despv = ComputeHog(samples, size);
	std::string modelfile = "model.yml";
	train_svm(despv, labels, modelfile);
	VehicleDetect(size, argv[2], modelfile);
  return 0;
}
