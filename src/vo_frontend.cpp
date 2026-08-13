#include "slam/vo_frontend.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

namespace slam {

VOFrontend::VOFrontend(const Eigen::Matrix3d& K, int max_features)
    : K_(K), max_features_(max_features) {
    orb_ = cv::ORB::create(max_features_, 1.2f, 8);
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}

void VOFrontend::reset() {
    prev_gray_ = cv::Mat();
    prev_kp_.clear();
    prev_desc_ = cv::Mat();
    last_inliers_ = 0;
}

void VOFrontend::filterMatches(
    const std::vector<std::vector<cv::DMatch>>& knn,
    std::vector<cv::DMatch>& good_matches) const {
    good_matches.clear();
    for (const auto& m : knn) {
        if (m.size() >= 2 && m[0].distance < 0.75 * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
}

void VOFrontend::toPointPairs(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) const {
    pts1.resize(matches.size());
    pts2.resize(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        pts1[i] = kp1[matches[i].queryIdx].pt;
        pts2[i] = kp2[matches[i].trainIdx].pt;
    }
}

SE3 VOFrontend::processFrame(const cv::Mat& frame) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Detect ORB features and compute descriptors
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    orb_->detectAndCompute(gray, cv::noArray(), kp, desc);

    if (prev_gray_.empty()) {
        // First frame: store and return identity
        prev_gray_ = gray.clone();
        prev_kp_ = kp;
        prev_desc_ = desc.clone();
        last_inliers_ = 0;
        return SE3();
    }

    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(prev_desc_, desc, knn, 2);

    std::vector<cv::DMatch> good_matches;
    filterMatches(knn, good_matches);

    if (good_matches.size() < 8) {
        // Not enough matches; treat as tracking lost
        std::cerr << "[VOFrontend] track lost: " << good_matches.size()
                  << " matches\n";
        prev_gray_ = gray.clone();
        prev_kp_ = kp;
        prev_desc_ = desc.clone();
        last_inliers_ = 0;
        return SE3();  // identity = no motion
    }

    // Extract matching point pairs
    std::vector<cv::Point2f> pts1, pts2;
    toPointPairs(prev_kp_, kp, good_matches, pts1, pts2);

    // Estimate essential matrix
    cv::Mat inlier_mask;
    cv::Mat K_cv = (cv::Mat_<double>(3, 3) << K_(0, 0), K_(0, 1), K_(0, 2),
                     K_(1, 0), K_(1, 1), K_(1, 2),
                     K_(2, 0), K_(2, 1), K_(2, 2));
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K_cv, cv::RANSAC, 0.999, 1.0, inlier_mask);

    // Count inliers
    int inliers = 0;
    if (!inlier_mask.empty()) {
        for (int i = 0; i < inlier_mask.rows; ++i)
            if (inlier_mask.at<uchar>(i)) ++inliers;
    }
    last_inliers_ = inliers;

    if (inliers < 6) {
        std::cerr << "[VOFrontend] too few inliers: " << inliers << "\n";
        prev_gray_ = gray.clone();
        prev_kp_ = kp;
        prev_desc_ = desc.clone();
        last_inliers_ = 0;
        return SE3();
    }

    // Recover relative pose (R, t) from the essential matrix
    cv::Mat R_cv, t_cv;
    cv::recoverPose(E, pts1, pts2, K_cv, R_cv, t_cv, inlier_mask);

    // Convert R (3x3) and t (3x1) from cv::Mat to Eigen
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = R_cv.at<double>(r, c);

    Eigen::Vector3d t;
    t(0) = t_cv.at<double>(0);
    t(1) = t_cv.at<double>(1);
    t(2) = t_cv.at<double>(2);

    // recoverPose gives T_12 = [R | t] (motion from frame 1 to frame 2 in
    // camera 1 frame).  Our delta_T for the EKF predict (T_{k+1} = T_k * delta_T)
    // is delta_T = T_12^{-1} = [R^T | -R^T * t].
    // t is unit-normalized (no metric scale).
    Eigen::Matrix3d Rt = R.transpose();
    Eigen::Vector3d tt = -Rt * t;
    SE3 delta_T(SO3(Rt), tt);

    // Store current frame for the next iteration
    prev_gray_ = gray.clone();
    prev_kp_ = kp;
    prev_desc_ = desc.clone();

    return delta_T;
}

} // namespace slam