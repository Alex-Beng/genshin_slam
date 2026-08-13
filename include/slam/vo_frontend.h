#ifndef SLAM_VO_FRONTEND_H
#define SLAM_VO_FRONTEND_H

#include "slam/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace slam {

// Monocular visual odometry frontend using ORB feature matching.
// Outputs relative SE(3) camera motion between consecutive frames.
// Translation scale is unit-normalized (no metric scale); the downstream
// EKF / graph corrects scale via minimap absolute observations.
class VOFrontend {
public:
    // K: camera intrinsic matrix (3x3)
    // max_features: max ORB features per frame
    VOFrontend(const Eigen::Matrix3d& K, int max_features = 2000);

    // Process a new image frame. Returns relative pose from the previous
    // frame to the current frame. For the first frame returns identity.
    SE3 processFrame(const cv::Mat& frame);

    // Number of inlier matches from the last estimate.
    int inlierCount() const { return last_inliers_; }

    // Reset the frontend state (e.g. after tracking lost).
    void reset();

private:
    Eigen::Matrix3d K_;
    int max_features_;

    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;

    cv::Mat prev_gray_;
    std::vector<cv::KeyPoint> prev_kp_;
    cv::Mat prev_desc_;

    int last_inliers_ = 0;

    // Filter matches by Lowe ratio test (< 0.75).
    void filterMatches(const std::vector<std::vector<cv::DMatch>>& knn,
                       std::vector<cv::DMatch>& good_matches) const;

    // Extract matched point coordinates from keypoints and matches.
    void toPointPairs(const std::vector<cv::KeyPoint>& kp1,
                      const std::vector<cv::KeyPoint>& kp2,
                      const std::vector<cv::DMatch>& matches,
                      std::vector<cv::Point2f>& pts1,
                      std::vector<cv::Point2f>& pts2) const;
};

} // namespace slam

#endif // SLAM_VO_FRONTEND_H