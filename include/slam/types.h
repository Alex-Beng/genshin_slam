#ifndef SLAM_TYPES_H
#define SLAM_TYPES_H

#include <opencv2/opencv.hpp>

namespace slam {

// 3x3 skew-symmetric matrix from 3x1 vector
inline cv::Mat skew(const cv::Mat& v) {
    CV_Assert(v.size() == cv::Size(1, 3) && v.type() == CV_64F);
    cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
    double v0 = v.at<double>(0);
    double v1 = v.at<double>(1);
    double v2 = v.at<double>(2);
    S.at<double>(0, 1) = -v2; S.at<double>(0, 2) =  v1;
    S.at<double>(1, 0) =  v2; S.at<double>(1, 2) = -v0;
    S.at<double>(2, 0) = -v1; S.at<double>(2, 1) =  v0;
    return S;
}

// vee operator: 3x3 skew-symmetric -> 3x1
inline cv::Mat vee(const cv::Mat& S) {
    CV_Assert(S.size() == cv::Size(3, 3) && S.type() == CV_64F);
    cv::Mat v(3, 1, CV_64F);
    v.at<double>(0) = S.at<double>(2, 1);
    v.at<double>(1) = S.at<double>(0, 2);
    v.at<double>(2) = S.at<double>(1, 0);
    return v;
}

// Make 4x4 transformation matrix from R (3x3) and t (3x1)
inline cv::Mat makeT(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t.copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

// Extract rotation from 4x4
inline cv::Mat rot(const cv::Mat& T) {
    return T(cv::Rect(0, 0, 3, 3)).clone();
}

// Extract translation from 4x4
inline cv::Mat trans(const cv::Mat& T) {
    return T(cv::Rect(3, 0, 1, 3)).clone();
}

} // namespace slam

#endif // SLAM_TYPES_H