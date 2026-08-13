#include "slam/minimap_odom.h"
#include <algorithm>
#include <cmath>

namespace slam {

MinimapOdom::MinimapOdom() {
    orb_ = cv::ORB::create(2000, 1.2f, 8);
}

void MinimapOdom::setReference(const cv::Mat& minimap) {
    ref_ = minimap.clone();
    have_ref_ = !ref_.empty();
}

bool MinimapOdom::orbOffset(const cv::Mat& a, const cv::Mat& b,
                            cv::Point2f& offset) const {
    cv::Mat gray_a, gray_b;
    if (a.channels() == 4) cv::cvtColor(a, gray_a, cv::COLOR_BGRA2GRAY);
    else if (a.channels() == 3) cv::cvtColor(a, gray_a, cv::COLOR_BGR2GRAY);
    else gray_a = a;
    if (b.channels() == 4) cv::cvtColor(b, gray_b, cv::COLOR_BGRA2GRAY);
    else if (b.channels() == 3) cv::cvtColor(b, gray_b, cv::COLOR_BGR2GRAY);
    else gray_b = b;

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;
    orb_->detectAndCompute(gray_a, cv::noArray(), kp1, d1);
    orb_->detectAndCompute(gray_b, cv::noArray(), kp2, d2);
    if (d1.empty() || d2.empty()) return false;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    std::vector<cv::DMatch> good;
    for (const auto& m : knn)
        if (m.size() >= 2 && m[0].distance < 0.6 * m[1].distance)
            good.push_back(m[0]);
    if (good.empty()) return false;

    cv::Point2f sum(0, 0);
    for (const auto& g : good)
        sum += kp2[g.trainIdx].pt - kp1[g.queryIdx].pt;
    offset = sum / (float)good.size();
    return true;
}

double MinimapOdom::rotationAngle(const cv::Mat& minimap, const cv::Mat& ref) const {
    cv::Mat a, b;
    if (minimap.channels() == 4) cv::cvtColor(minimap, a, cv::COLOR_BGRA2GRAY);
    else if (minimap.channels() == 3) cv::cvtColor(minimap, a, cv::COLOR_BGR2GRAY);
    else a = minimap;
    if (ref.channels() == 4) cv::cvtColor(ref, b, cv::COLOR_BGRA2GRAY);
    else if (ref.channels() == 3) cv::cvtColor(ref, b, cv::COLOR_BGR2GRAY);
    else b = ref;

    if (a.size() != b.size() || a.empty()) return 0.0;

    // Polar-warp both minimaps; angular displacement becomes a translation
    // along the polar image's columns (circle of radius R).
    const int CW = 512;
    const double R = std::max(a.cols, a.rows) * 0.5;
    cv::Mat pa, pb;
    cv::warpPolar(a, pa, cv::Size(CW, CW), cv::Point2f(a.cols / 2.0, a.rows / 2.0),
                  R, cv::WARP_POLAR_LINEAR);
    cv::warpPolar(b, pb, cv::Size(CW, CW), cv::Point2f(b.cols / 2.0, b.rows / 2.0),
                  R, cv::WARP_POLAR_LINEAR);

    // Use the annular band (rows 25%..75%) as a 2-D signature:
    // rows = radius, cols = angle. A pure rotation shifts cols.
    cv::Mat sA = pa.rowRange(CW / 4, 3 * CW / 4).clone();
    cv::Mat sB = pb.rowRange(CW / 4, 3 * CW / 4).clone();
    if (sA.empty() || sB.empty()) return 0.0;
    sA.convertTo(sA, CV_32F);
    sB.convertTo(sB, CV_32F);

    // Phase correlate along the angle axis (treat columns as periodic);
    // the returned shift.x is in pixels of the polar image width.
    cv::Point2d shift = cv::phaseCorrelate(sB, sA);
    double ang = shift.x * 2.0 * CV_PI / (double)CW;
    if (ang > CV_PI) ang -= 2.0 * CV_PI;
    if (ang < -CV_PI) ang += 2.0 * CV_PI;
    return ang;
}

MinimapOdomResult MinimapOdom::odometry(const cv::Mat& minimap) {
    MinimapOdomResult res;
    if (!have_ref_ || ref_.empty() || minimap.empty()) {
        setReference(minimap);
        return res;
    }

    cv::Point2f off;
    if (orbOffset(ref_, minimap, off)) {
        res.dx_px = off.x;
        res.dz_px = off.y;
    }

    res.dyaw = rotationAngle(minimap, ref_);
    res.valid = true;
    setReference(minimap);
    return res;
}

} // namespace slam