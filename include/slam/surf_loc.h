#ifndef SLAM_SURF_LOC_H
#define SLAM_SURF_LOC_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

namespace slam {

// Locates the minimap within the big map via SURF feature matching.
// Low-frequency absolute global pose estimation (x, z) in big-map pixel space.
class SurfLocator {
public:
    SurfLocator();

    // Set the big map feature cache (loaded from map_store). Builds FLANN index.
    bool setMapFeatures(const std::vector<cv::KeyPoint>& map_kp,
                        const cv::Mat& map_desc);

    // Locate minimap in big map. Returns false if match is unreliable.
    // out_pt: big-map working-pixel coordinates (x = columns, z = rows).
    // out_scale: fitted per-axis scale (map px per minimap px); used to convert
    //   minimap-pixel odometry into map-pixel units.
    bool locate(const cv::Mat& minimap, cv::Point2d& out_pt,
                double& out_scale, int& inliers, double& score);

    // Continuity-friendly variant: search within a neighbourhood rect of a
    // previous estimate. smaller search => faster and more robust.
    bool locateNear(const cv::Mat& minimap, const cv::Point2d& prev_pt,
                    double half_window, cv::Point2d& out_pt,
                    double& out_scale, int& inliers, double& score);

private:
    cv::Ptr<cv::xfeatures2d::SURF> surf_;
    std::vector<cv::KeyPoint> map_kp_;
    cv::Mat map_desc_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    bool ready_ = false;
    double scale_factor_ = 1.0;

    bool locateImpl(const cv::Mat& minimap, const cv::Mat& search_img,
                    bool has_search, cv::Point2d& out_pt,
                    double& out_scale, int& inliers, double& score);
};

} // namespace slam

#endif // SLAM_SURF_LOC_H