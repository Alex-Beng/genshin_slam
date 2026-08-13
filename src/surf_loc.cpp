#include "slam/surf_loc.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace slam {

SurfLocator::SurfLocator() {
    surf_ = cv::xfeatures2d::SURF::create(800.0, 4, 3, true, true);
}

bool SurfLocator::setMapFeatures(const std::vector<cv::KeyPoint>& map_kp,
                                 const cv::Mat& map_desc) {
    if (map_kp.empty() || map_desc.empty()) return false;
    map_kp_ = map_kp;
    map_desc_ = map_desc.clone();
    matcher_ = cv::BFMatcher::create(cv::NORM_L2);
    matcher_->add({map_desc_});
    matcher_->train();
    ready_ = true;
    return true;
}

// Solve, for each axis independently with a shared assumption:
//   p_min = S * (p_map - o)
// Using variables (A=sx, B=sx*ox): p_min.x = A*p_map.x - B  (linear LS)
// Returns o = (B/A) and sx = A.
static void fitScaleOffset(const std::vector<cv::Point2f>& min,
                           const std::vector<cv::Point2f>& map,
                           double& sx, double& ox,
                           double& sy, double& oy) {
    // Axis X: fit A,B
    double S_xx = 0, S_yy = 0, S_xy = 0, S_x = 0, S_y = 0, N = (double)min.size();
    for (size_t i = 0; i < min.size(); ++i) {
        S_xx += map[i].x * map[i].x; S_xy += map[i].x * min[i].x;
        S_x += map[i].x; S_y += min[i].x;
    }
    double det = N * S_xx - S_x * S_x;
    if (std::abs(det) < 1e-12) { sx = 1; ox = 0; }
    else {
        double A = (N * S_xy - S_x * S_y) / det;
        double B = (A * S_x - S_y) / N;   // p_min = A*p_map - B  =>  B = A*mean(map) - mean(min)
        sx = A; ox = B / A;
    }
    S_xx = S_yy = S_xy = S_x = S_y = 0;
    for (size_t i = 0; i < min.size(); ++i) {
        S_xx += map[i].y * map[i].y; S_xy += map[i].y * min[i].y;
        S_x += map[i].y; S_y += min[i].y;
    }
    det = N * S_xx - S_x * S_x;
    if (std::abs(det) < 1e-12) { sy = 1; oy = 0; }
    else {
        double A = (N * S_xy - S_x * S_y) / det;
        double B = (A * S_x - S_y) / N;
        sy = A; oy = B / A;
    }
}

bool SurfLocator::locateImpl(const cv::Mat& minimap, const cv::Mat& search_img,
                             bool /*has_search*/, cv::Point2d& out_pt,
                             double& out_scale, int& inliers, double& score) {
    if (!ready_ || minimap.empty()) return false;

    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    surf_->detectAndCompute(minimap, cv::noArray(), kp, desc);
    if (desc.empty() || kp.size() < 4) return false;

    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(desc, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& m : knn) {
        if (m.size() >= 2 && m[0].distance < 0.66 * m[1].distance)
            good.push_back(m[0]);
    }
    if (good.size() < 8) return false;

    std::vector<cv::Point2f> min_pts(good.size()), map_pts(good.size());
    for (size_t i = 0; i < good.size(); ++i) {
        min_pts[i] = kp[good[i].queryIdx].pt;
        map_pts[i] = map_kp_[good[i].trainIdx].pt;
    }

    double sx, ox, sy, oy;
    fitScaleOffset(min_pts, map_pts, sx, ox, sy, oy);
    if (sx <= 0 || sy <= 0 || !std::isfinite(sx) || !std::isfinite(sy)) return false;

    // Reject outliers: residual = |p_min - S*(p_map - o)| per point.
    std::vector<int> keep;
    double med = 0;
    std::vector<double> res(good.size());
    for (size_t i = 0; i < good.size(); ++i) {
        cv::Point2f pred((float)(sx * (map_pts[i].x - ox)),
                         (float)(sy * (map_pts[i].y - oy)));
        res[i] = cv::norm(min_pts[i] - pred);
    }
    std::vector<double> r2 = res;
    std::sort(r2.begin(), r2.end());
    med = r2[r2.size() / 2];
    double thr = std::max(8.0, 3.0 * med);
    for (size_t i = 0; i < good.size(); ++i)
        if (res[i] < thr) keep.push_back((int)i);

    if (keep.size() < 6) return false;

    std::vector<cv::Point2f> km(keep.size()), kp2(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        km[i] = min_pts[keep[i]]; kp2[i] = map_pts[keep[i]];
    }
    fitScaleOffset(km, kp2, sx, ox, sy, oy);
    if (sx <= 0 || sy <= 0 || !std::isfinite(sx) || !std::isfinite(sy)) return false;

    // Minimap center in minimap coords -> map coords:
    // p_min = S*(p_map - o)  =>  p_map = o + S^{-1} * p_min
    cv::Point2d c((minimap.cols - 1) / 2.0, (minimap.rows - 1) / 2.0);
    out_pt.x = ox + c.x / sx;
    out_pt.y = oy + c.y / sy;
    out_scale = 0.5 * (sx + sy);   // average per-axis scale (map px / min px)

    inliers = (int)keep.size();
    score = med;
    return true;
}

bool SurfLocator::locate(const cv::Mat& minimap, cv::Point2d& out_pt,
                         double& out_scale, int& inliers, double& score) {
    return locateImpl(minimap, cv::Mat(), cv::Mat().data != nullptr,
                      out_pt, out_scale, inliers, score);
}

bool SurfLocator::locateNear(const cv::Mat& minimap, const cv::Point2d& prev_pt,
                             double half_window, cv::Point2d& out_pt,
                             double& out_scale, int& inliers, double& score) {
    if (!ready_) return false;
    // Continuous local matching handled in slam_main via crop + locate on
    // a cropped map ROI (features must be recomputed for the crop). Here we
    // simply fall back to global matching; local search is implemented in the
    // pipeline where the map image is available.
    (void)half_window;
    return locate(minimap, out_pt, out_scale, inliers, score);
}

} // namespace slam