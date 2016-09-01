#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <boost/filesystem.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;
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

void buildVoc(std::vector<cv::Mat>& trainImage, cv::Mat& vocabulary) {
  int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	Ptr<SURF> extractor = SURF::create();
	vector<KeyPoint> keypoints;	
	cv::Mat descriptors;
	cv::Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	for (const auto& img : trainImage) {
		detector->detect(img,keypoints);
		extractor->compute(img, keypoints, descriptors);
		training_descriptors.push_back(descriptors);
	}
	BOWKMeansTrainer bowtrainer(1000); //num clusters
	bowtrainer.add(training_descriptors);
	vocabulary = bowtrainer.cluster();
}

void  trainBovw(std::vector<cv::Mat>& trainImage) {
	cv::Mat vocabulary;
	buildVoc(trainImage,vocabulary);
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
	Ptr<SURF> extractor = SURF::create();
	Ptr<DescriptorMatcher> matcher = new cv::FlannBasedMatcher{};
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	map<string,Mat> classes_training_data; classes_training_data.clear();
}
