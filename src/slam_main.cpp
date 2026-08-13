#include "slam/capture.h"
#include "slam/minimap_pipeline.h"
#include "slam/map_store.h"
#include "slam/surf_loc.h"
#include "slam/minimap_odom.h"
#include "slam/ekf_slam.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace slam;

int main(int argc, char** argv) {
    const std::string video =
        argc > 1 ? argv[1]
                 : "F:/Backup/录屏/原神 2022-01-03 11-14-53.mp4";
    const std::string map_path =
        argc > 2 ? argv[2] : "C:/Users/Alex Beng/Downloads/output.png";
    const std::string cache_path =
        argc > 3 ? argv[3] : "bigmap_features.bin";

    // ---------- MapStore: big map minus map scale for pixel->world ----------
    // World coords = map pixel coords (output.png pixels). SURF locator works
    // in downscaled map pixels; convert back to original map pixels.
    MapStore store;
    if (!store.load(map_path, 4.0)) {
        std::cerr << "Failed to load map\n";
        return 1;
    }
    if (!store.loadCache(cache_path)) {
        std::cout << "Building map features...\n";
        store.buildFeatures(1024, 128, 800.0);
        if (!store.saveCache(cache_path))
            std::cerr << "cache save failed\n";
    }
    std::cout << "Big map features: " << store.keypoints().size() << "\n";
    const double map_scale_inv = 1.0 / store.scale();  // working->original px

    // ---------- Surf locator ----------
    SurfLocator locator;
    if (!locator.setMapFeatures(store.keypoints(), store.descriptors())) {
        std::cerr << "Failed to init locator\n";
        return 1;
    }

    // ---------- Capture ----------
    VideoFileCapture cap(video);
    if (!cap.init()) {
        std::cerr << "Could not open " << video << "\n";
        return 1;
    }

    MinimapPipeline mm;
    MinimapOdom odom;
    bool have_prev = false;
    cv::Point2d last_pos(-1, -1);  // big map original px

    // ---------- EKF ----------
    EKFSLAM ekf;
    const double px_per_m = 2.557;   // map px per world unit (GT meters ~)
    // start at the first SURF localization if available; else origin, with
    // large initial uncertainty so the first absolute obs dominates
    Matrix12 P0 = Matrix12::Zero();
    for (int i = 0; i < 3; ++i) P0(i, i) = 1e6;
    for (int i = 3; i < 6; ++i) P0(i, i) = 1.0;
    for (int i = 6; i < 12; ++i) P0(i, i) = 1e-6;
    ekf.init(SE3(), SE3(), P0);
    Matrix12 Q = Matrix12::Zero();
    for (int i = 0; i < 3; ++i) Q(i, i) = 0.02;
    for (int i = 3; i < 6; ++i) Q(i, i) = 0.01;
    for (int i = 6; i < 12; ++i) Q(i, i) = 1e-6;

    Matrix3 R_abs = Matrix3::Zero();
    R_abs(0, 0) = 1.0 / (store.scale() * store.scale());   // ~4px work
    R_abs(1, 1) = 1.0 / (store.scale() * store.scale());
    R_abs(2, 2) = 0.1;
    Matrix3 R_odom = Matrix3::Zero();
    R_odom(0, 0) = 0.1; R_odom(1, 1) = 0.1; R_odom(2, 2) = 0.05;

    // ---------- Main loop ----------
    cv::Mat frame, minimap;
    cv::Rect mrect;
    int idx = 0, n_locate = 0, n_odom = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        ++idx;
        if (!mm.extract(frame, minimap, mrect)) continue;

        // Absolute localization every ~10 frames
        cv::Point2d map_pt;
        double map_scale = 1.0;   // map working px per minimap px
        int inl = 0; double sc = 0;
        if (idx % 10 == 1) {
            bool ok = locator.locate(minimap, map_pt, map_scale, inl, sc);
            if (ok) {
                last_pos = map_pt;      // working px
                ++n_locate;
                // world coords in working-map pixels
                MapObs obs;
                obs << map_pt.x / px_per_m, map_pt.y / px_per_m, 0.0;
                ekf.update(obs, R_abs);
            }
        }

        // Relative odometry each frame
        if (!have_prev) {
            odom.setReference(minimap);
            have_prev = true;
        } else {
            MinimapOdomResult res = odom.odometry(minimap);
            if (res.valid) {
                ++n_odom;
                // convert minimap px -> map working px using SURF-derived scale
                double s = map_scale;
                Eigen::Vector3d d;
                d << res.dx_px * s / px_per_m, res.dz_px * s / px_per_m, res.dyaw;
                ekf.updateMinimapOdom(d, R_odom);
            }
        }

        if (idx % 60 == 0) {
            SE3 cam = ekf.getCameraPose();
            Eigen::Vector3d t = cam.translation();
            std::cout << "frame " << std::setw(4) << idx
                      << " | SURF " << n_locate << " odom " << n_odom
                      << " | pos(px) x=" << std::setprecision(2)
                      << (last_pos.x > 0 ? last_pos.x * map_scale_inv : 0)
                      << " z=" << (last_pos.y > 0 ? last_pos.y * map_scale_inv : 0)
                      << " | EKF t=(" << t(0) << ", " << t(2) << ")"
                      << " inl=" << inl << " scale=" << map_scale << "\n";
        }
    }
    double sec = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nProcessed " << idx << " frames in " << sec << "s\n";
    std::cout << "SURF localizations: " << n_locate
              << ", odometry updates: " << n_odom << "\n";

    cap.release();
    return 0;
}