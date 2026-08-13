#include "slam/map_store.h"
#include "slam/surf_loc.h"
#include <iostream>
#include <random>

using namespace slam;

int main(int argc, char** argv) {
    const std::string map_path =
        argc > 1 ? argv[1] : "C:/Users/Alex Beng/Downloads/output.png";
    const std::string cache_path =
        argc > 2 ? argv[2] : "bigmap_features.bin";

    MapStore store;
    if (!store.load(map_path, 4.0)) { std::cerr << "load fail\n"; return 1; }
    if (!store.loadCache(cache_path)) {
        store.buildFeatures(1024, 128, 800.0);
        store.saveCache(cache_path);
    }
    std::cout << "features: " << store.keypoints().size() << "\n";

    SurfLocator loc;
    if (!loc.setMapFeatures(store.keypoints(), store.descriptors())) {
        std::cerr << "locator init fail\n";
        return 1;
    }

    // Simulate minimaps: crop 200x200 patches from random map locations
    // (in working-pixel coords), then locate them back.
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dx(500, store.size().width - 700);
    std::uniform_int_distribution<int> dy(500, store.size().height - 700);

    const double scale_inv = 1.0 / store.scale();
    int ok_count = 0, total = 10;
    for (int t = 0; t < total; ++t) {
        int gx = dx(rng), gy = dy(rng);
        cv::Mat patch = store.crop(gx - 100, gy - 100, 200, 200);
        if (patch.empty()) continue;

        cv::Point2d est;
        double scfac = 1.0;
        int inl = 0; double sc = 0;
        bool ok = loc.locate(patch, est, scfac, inl, sc);
        // print working-pixel estimate too
        std::cout << "  work_px est=(" << est.x << "," << est.y
                  << ") gt_work=(" << gx << "," << gy
                  << ") scale=" << scfac << "\n";

        // ground truth working px center in original map px
        double gx_orig = gx * scale_inv, gy_orig = gy * scale_inv;
        cv::Point2d est_orig(est.x * scale_inv, est.y * scale_inv);
        double err = cv::norm(cv::Point2d(gx_orig, gy_orig) - est_orig);
        std::cout << "t" << t << " gt=(" << (int)gx_orig << "," << (int)gy_orig
                  << ") est=(" << (int)est_orig.x << "," << (int)est_orig.y
                  << ") ok=" << ok << " inl=" << inl << " err=" << (int)err
                  << "px\n";
        if (ok && err < 100.0) ++ok_count;
    }
    std::cout << "OK " << ok_count << "/" << total << "\n";
    return 0;
}