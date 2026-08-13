#ifndef SLAM_MAP_STORE_H
#define SLAM_MAP_STORE_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace slam {

// Loads the stitched big map image, downscales to a working gray image, and
// precomputes SURF features block-by-block (with boundary padding; features
// outside each block's valid region are discarded). Features can be cached to
// disk to avoid recomputation on later runs.
class MapStore {
public:
    MapStore();

    // Load map from PNG (any type; converted to downscaled gray).
    bool load(const std::string& path, double downsample);

    // Build SURF features block-by-block.
    void buildFeatures(int block_size = 1024, int pad = 128,
                       double hessian = 800.0);

    // Cache keypoints + descriptors to disk (custom binary format).
    bool saveCache(const std::string& path) const;
    bool loadCache(const std::string& path);

    // Map size in working (downscaled) pixels.
    cv::Size size() const;

    // Crop a region from the working gray map (clamped). Return empty if no overlap.
    cv::Mat crop(int x, int y, int w, int h) const;

    // Gray working map image (downscaled). For visualization only.
    const cv::Mat& gray() const { return gray_; }

    // Scale factor from original map pixels to working pixels.
    double scale() const { return scale_; }

    const std::vector<cv::KeyPoint>& keypoints() const { return kp_; }
    const cv::Mat& descriptors() const { return desc_; }

private:
    cv::Mat gray_;      // downscaled gray map (uint8)
    double scale_ = 1.0;
    bool map_loaded_ = false;

    std::vector<cv::KeyPoint> kp_;
    cv::Mat desc_;
    bool features_ready_ = false;
};

} // namespace slam

#endif // SLAM_MAP_STORE_H