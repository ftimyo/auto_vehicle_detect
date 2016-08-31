#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
int main(int, char* argv[2]) {
	cv::Mat img_object = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img_scene = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	vector<cv::KeyPoint> kps1,kps2;
	detector->detect(img_object,kps1);
	detector->detect(img_scene,kps2);
	std::cout << kps1.size() << ',' << kps2.size() << std::endl;
	Mat dsp1,dsp2;
	Ptr<SURF> extractor = SURF::create();
	extractor->compute(img_object,kps1,dsp1);
	extractor->compute(img_scene,kps2,dsp2);
	cv::FlannBasedMatcher matcher;
	vector<cv::DMatch> matches;
	matcher.match(dsp1,dsp2,matches);

  double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < dsp1.rows; i++) {
		double dist = matches[i].distance;
    if(dist < min_dist) min_dist = dist;
    if(dist > max_dist) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector<cv::DMatch> good_matches;
	for(int i = 0; i < dsp1.rows; i++) {
		//std::cout << matches[i].distance <<" | "<< min_dist <<" | "<<max_dist<< std::endl;
		if (matches[i].distance < 3*min_dist) {
			good_matches.push_back(matches[i]);
		}
  }
  Mat img_matches;
  drawMatches( img_object, kps1, img_scene, kps2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#if 1
  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for(size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( kps1[good_matches[i].queryIdx].pt );
    scene.push_back( kps2[good_matches[i].trainIdx].pt );
  }
	//std::cout << obj.size() << std::endl;	
	//std::cout << scene.size() << std::endl;	

  Mat H = findHomography(obj, scene,cv::RANSAC);
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0);
	obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
	obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);
	perspectiveTransform( obj_corners, scene_corners, H);
	/*
  line( img_scene, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
  line( img_scene, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
  line( img_scene, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
  line( img_scene, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
	*/
	cv::Rect bb = cv::boundingRect(scene_corners);
	cv::rectangle(img_scene,bb,Scalar(0,255,0),4);
	imshow("x",img_scene);
#endif
	//imshow("x",img_matches);
	cv::waitKey(0);
	return 0;
}
