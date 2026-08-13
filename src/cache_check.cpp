#include "slam/map_store.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace slam;

int main() {
    MapStore store;
    if (!store.load("C:/Users/Alex Beng/Downloads/output.png", 4.0)) return 1;
    if (!store.loadCache("bigmap_features.bin")) {
        std::cout << "cache missing; build\n";
        store.buildFeatures(1024, 128, 800.0);
        store.saveCache("bigmap_features.bin");
    }
    int n = (int)store.keypoints().size();
    std::cout << "cache kp=" << n << " desc=" << store.descriptors().size() << "\n";

    // Verify alignment: detect SURF near a cached keypoint and check descriptor distance.
    cv::Ptr<cv::xfeatures2d::SURF> surf =
        cv::xfeatures2d::SURF::create(800.0, 4, 3, true, true);
    int tested = 0, aligned = 0;
    for (int i = 0; i < n && tested < 20; i += n / 20) {
        cv::KeyPoint k = store.keypoints()[i];
        cv::Mat win = store.crop((int)k.pt.x - 20, (int)k.pt.y - 20, 40, 40);
        if (win.empty()) continue;
        std::vector<cv::KeyPoint> kp;
        cv::Mat d;
        surf->detectAndCompute(win, cv::noArray(), kp, d);
        if (d.rows == 0) continue;
        ++tested;
        // nearest cached descriptor to this win's first descriptor
        cv::BFMatcher m(cv::NORM_L2);
        std::vector<cv::DMatch> mm;
        m.match(d, store.descriptors(), mm);
        // best match index
        int bidx = 0;
        for (int j = 1; j < (int)mm.size(); ++j)
            if (mm[j].distance < mm[bidx].distance) bidx = j;
        cv::KeyPoint bk = store.keypoints()[mm[bidx].trainIdx];
        double dist = cv::norm(cv::Point2f(bk.pt.x, bk.pt.y) -
                               cv::Point2f(k.pt.x, k.pt.y));
        if (dist < 5.0) ++aligned;
        std::cout << "kp#"<<i<<" at ("<<(int)k.pt.x<<","<<(int)k.pt.y
                  <<") best-match kp#"<<mm[bidx].trainIdx<<" at ("
                  <<(int)bk.pt.x<<","<<(int)bk.pt.y<<") dist="<<(int)dist
                  <<" dmin="<<(int)mm[bidx].distance<<"\n";
    }
    std::cout << "aligned " << aligned << "/" << tested << "\n";
    return 0;
}