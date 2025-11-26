#include "src/Stag.h"
#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

int main() {
  // load image
  cv::Mat image = cv::imread("../example.jpg");

  // !仅检测指定的tag family
  // set HD library
  int libraryHD = 21;

  auto corners = std::vector<std::vector<cv::Point2f>>();
  auto ids = std::vector<int>();
  auto rejectedImgPoints =
      std::vector<std::vector<cv::Point2f>>(); // optional, helpful for
                                               // debugging

  // detect markers
  auto t0 = std::chrono::high_resolution_clock::now();
  stag::detectMarkers(image, libraryHD, corners, ids, rejectedImgPoints);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout
      << "STag detection time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << " ms" << std::endl;

  // draw and save results
  stag::drawDetectedMarkers(image, corners, ids);
  cv::imwrite("example_result.jpg", image);

  // *这里的相机内参、畸变系数、边长等仅用于验证接口
  cv::Mat cameraMatrix =
      (cv::Mat_<float>(3, 3) << 960, 0, 640, 0, 640, 320, 0, 0, 1);
  cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_32F);

  // !黑色部分的边长
  constexpr float edge_length = 0.16f;

  std::size_t N = corners.size();
  for (std::size_t i = 0; i < N; ++i) {
    std::cout << "====================================" << std::endl;
    std::cout << "Marker " << ids[i] << " detected with " << corners[i].size()
              << " corners" << std::endl;
    if (corners[i].size() != 4) {
      continue;
    }
    // !STag的坐标系原点定义在左上角，stag::detectMarkers检测出的角点是有序的
    // !从原点开始，俯视图下逆时针方向，世界坐标系中的角点坐标顺序需要与像素坐标系中的角点坐标顺序一致
    std::vector<cv::Point3f> Pw;
    Pw.emplace_back(cv::Point3f{0, 0, 0});
    Pw.emplace_back(cv::Point3f{0, edge_length, 0});
    Pw.emplace_back(cv::Point3f{edge_length, edge_length, 0});
    Pw.emplace_back(cv::Point3f{edge_length, 0, 0});
    std::vector<cv::Point2f> Puv;
    for (std::size_t j = 0; j < corners[i].size(); ++j) {
      std::cout << "Corner " << j << " at " << corners[i][j] << std::endl;
      Puv.emplace_back(corners[i][j]);
    }
    // !Pc = Rcw * Pw + tcw
    cv::Mat rvec, tvec;
    cv::solvePnP(Pw, Puv, cameraMatrix, distCoeffs, rvec, tvec);
    std::cout << "rvec: " << rvec << std::endl;
    std::cout << "tvec: " << tvec << std::endl;
    // todo：测试这个接口
    // cv::solvePnPRansac(Pw, Puv, cameraMatrix, distCoeffs, rvec, tvec);
    // std::cout << "rvec: " << rvec << std::endl;
    // std::cout << "tvec: " << tvec << std::endl;
  }

  return 0;
}