#ifndef SLAM_MINIMAP_PIPELINE_H
#define SLAM_MINIMAP_PIPELINE_H

#include <opencv2/opencv.hpp>

namespace slam {

// Extracts the in-game minimap (Genshin top-left circular mini-map) from a
// full game frame. The rect is derived from the 1920x1080 reference layout
// (center (168,124), radius ~107) and scaled proportionally for other sizes.
class MinimapPipeline {
public:
    MinimapPipeline();

    // Set reference minimap circle (fraction) for non-1080p frames:
    // cx, cy, radius given in fractions of width/height relative to the
    // 1920x1080 reference. By default uses the Genshin reference constants.
    void setReference(double frac_cx, double frac_cy, double frac_r);

    // Crop the minimap square (diameter = 2*r) from the full frame.
    // Output is a square containing the minimap disk.
    bool extract(const cv::Mat& screen, cv::Mat& minimap, cv::Rect& rect) const;

    // Full-frame coordinates of the minimap rect (last extract()).
    const cv::Rect& lastRect() const { return last_rect_; }

private:
    double frac_cx_, frac_cy_, frac_r_;
    mutable cv::Rect last_rect_;
};

} // namespace slam

#endif // SLAM_MINIMAP_PIPELINE_H