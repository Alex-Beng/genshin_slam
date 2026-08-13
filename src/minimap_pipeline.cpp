#include "slam/minimap_pipeline.h"
#include <cmath>

namespace slam {

// Genshin 1920x1080 reference: MiniMapRect {59,15,218,218} -> center (168,124),
// radius ~107.
MinimapPipeline::MinimapPipeline() {
    setReference(168.0 / 1920.0, 124.0 / 1080.0, 107.0 / 1080.0);
}

void MinimapPipeline::setReference(double frac_cx, double frac_cy, double frac_r) {
    frac_cx_ = frac_cx;
    frac_cy_ = frac_cy;
    frac_r_ = frac_r;
}

bool MinimapPipeline::extract(const cv::Mat& screen, cv::Mat& minimap,
                              cv::Rect& rect) const {
    if (screen.empty()) return false;
    const int w = screen.cols;
    const int h = screen.rows;

    const double cx = frac_cx_ * w;
    const double cy = frac_cy_ * h;
    const double r = frac_r_ * h;

    const int side = (int)std::ceil(2.0 * r);
    const int x0 = (int)std::round(cx - r);
    const int y0 = (int)std::round(cy - r);

    rect = cv::Rect(x0, y0, side, side);
    // Clamp to frame
    rect = rect & cv::Rect(0, 0, w, h);
    if (rect.empty()) return false;

    minimap = screen(rect).clone();
    last_rect_ = rect;
    return true;
}

} // namespace slam