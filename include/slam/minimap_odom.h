#ifndef SLAM_MINIMAP_ODOM_H
#define SLAM_MINIMAP_ODOM_H

#include <opencv2/opencv.hpp>

namespace slam {

// Result of minimap inter-frame odometry.
struct MinimapOdomResult {
    bool valid = false;
    double dx_px = 0.0;    // translation in minimap pixels (x)
    double dz_px = 0.0;    // translation in minimap pixels (z = image y)
    double dyaw = 0.0;     // rotation in radians
};

// Computes minimap inter-frame odometry:
//  - translation (dx, dz) via ORB feature matching between two minimap images
//  - rotation dyaw via polar-warped edge analysis (Genshin minimap is a disk)
class MinimapOdom {
public:
    MinimapOdom();

    // Feed the previous minimap image as the new reference.
    void setReference(const cv::Mat& minimap);

    // Compute odometry from the reference (prev) to `minimap`.
    MinimapOdomResult odometry(const cv::Mat& minimap);

private:
    cv::Mat ref_;
    bool have_ref_ = false;
    cv::Ptr<cv::ORB> orb_;

    bool orbOffset(const cv::Mat& a, const cv::Mat& b, cv::Point2f& offset) const;
    double rotationAngle(const cv::Mat& minimap, const cv::Mat& ref) const;
};

} // namespace slam

#endif // SLAM_MINIMAP_ODOM_H