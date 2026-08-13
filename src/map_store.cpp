#include "slam/map_store.h"
#include <opencv2/xfeatures2d.hpp>
#include <fstream>

namespace slam {

MapStore::MapStore() {}

bool MapStore::load(const std::string& path, double downsample) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return false;

    if (img.depth() != CV_8U) img.convertTo(img, CV_8U, 1.0 / (img.depth() == CV_16U ? 257.0 : 1.0));
    if (img.channels() == 4) cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    scale_ = downsample > 1.0 ? 1.0 / downsample : 1.0;
    if (downsample > 1.0)
        cv::resize(img, gray_, cv::Size(), 1.0 / downsample, 1.0 / downsample,
                   cv::INTER_AREA);
    else
        gray_ = img;

    map_loaded_ = true;
    features_ready_ = false;
    kp_.clear();
    desc_.release();
    return true;
}

cv::Size MapStore::size() const { return gray_.size(); }

cv::Mat MapStore::crop(int x, int y, int w, int h) const {
    cv::Rect r(x, y, w, h);
    r = r & cv::Rect(0, 0, gray_.cols, gray_.rows);
    if (r.empty()) return cv::Mat();
    return gray_(r).clone();
}

void MapStore::buildFeatures(int block_size, int pad, double hessian) {
    if (!map_loaded_) return;

    const int W = gray_.cols, H = gray_.rows;
    kp_.clear();
    desc_.release();
    std::vector<cv::KeyPoint> all_kp;
    std::vector<cv::Mat> all_desc;

    cv::Ptr<cv::xfeatures2d::SURF> surf =
        cv::xfeatures2d::SURF::create(hessian, 4, 3, true, true);

    for (int by = 0; by < H; by += block_size) {
        for (int bx = 0; bx < W; bx += block_size) {
            cv::Rect block(bx, by, block_size, block_size);
            cv::Rect padded = block;
            padded.x -= pad; padded.y -= pad;
            padded.width += 2 * pad; padded.height += 2 * pad;
            padded &= cv::Rect(0, 0, W, H);
            if (padded.empty()) continue;

            cv::Mat tile = gray_(padded);
            std::vector<cv::KeyPoint> kp;
            cv::Mat desc;
            surf->detectAndCompute(tile, cv::noArray(), kp, desc);
            if (desc.empty()) continue;

            // Keep only keypoints within the valid (unpadded) block, mapped
            // back to global working-map coordinates.
            std::vector<cv::KeyPoint> keep;
            std::vector<int> keep_idx;
            for (int i = 0; i < (int)kp.size(); ++i) {
                cv::KeyPoint k = kp[i];
                k.pt.x += padded.x;
                k.pt.y += padded.y;
                if (block.contains(k.pt)) {
                    keep.push_back(k);
                    keep_idx.push_back(i);
                }
            }
            if (keep.empty()) continue;

            cv::Mat desc_sub(keep.size(), desc.cols, desc.type());
            for (size_t i = 0; i < keep.size(); ++i)
                desc.row(keep_idx[i]).copyTo(desc_sub.row((int)i));

            all_kp.insert(all_kp.end(), keep.begin(), keep.end());
            all_desc.push_back(desc_sub);
        }
    }

    if (all_desc.empty()) return;
    cv::vconcat(all_desc, desc_);
    kp_ = std::move(all_kp);
    features_ready_ = true;
}

bool MapStore::saveCache(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    int n = (int)kp_.size();
    ofs.write((const char*)&n, sizeof(n));
    for (const auto& k : kp_) {
        float x = k.pt.x, y = k.pt.y, s = k.size, a = k.angle,
              r = k.response, o = (float)k.octave;
        ofs.write((const char*)&x, 4); ofs.write((const char*)&y, 4);
        ofs.write((const char*)&s, 4); ofs.write((const char*)&a, 4);
        ofs.write((const char*)&r, 4); ofs.write((const char*)&o, 4);
    }
    int rows = desc_.rows, cols = desc_.cols, type = desc_.type();
    ofs.write((const char*)&rows, 4); ofs.write((const char*)&cols, 4);
    ofs.write((const char*)&type, 4);
    ofs.write((const char*)desc_.data, (std::streamsize)desc_.total() * desc_.elemSize());
    return true;
}

bool MapStore::loadCache(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    int n = 0;
    ifs.read((char*)&n, sizeof(n));
    if (n <= 0) return false;
    kp_.resize(n);
    for (int i = 0; i < n; ++i) {
        float x, y, s, a, r, o;
        ifs.read((char*)&x, 4); ifs.read((char*)&y, 4);
        ifs.read((char*)&s, 4); ifs.read((char*)&a, 4);
        ifs.read((char*)&r, 4); ifs.read((char*)&o, 4);
        kp_[i].pt = cv::Point2f(x, y);
        kp_[i].size = s; kp_[i].angle = a;
        kp_[i].response = r; kp_[i].octave = (int)o;
    }
    int rows, cols, type;
    ifs.read((char*)&rows, 4); ifs.read((char*)&cols, 4); ifs.read((char*)&type, 4);
    desc_ = cv::Mat(rows, cols, type);
    ifs.read((char*)desc_.data, (std::streamsize)desc_.total() * desc_.elemSize());
    features_ready_ = !desc_.empty();
    return true;
}

} // namespace slam