#include "slam/map_store.h"
#include <iostream>
#include <chrono>

using namespace slam;

int main(int argc, char** argv) {
    const std::string map_path =
        argc > 1 ? argv[1] : "C:/Users/Alex Beng/Downloads/output.png";
    const std::string cache_path =
        argc > 2 ? argv[2] : "bigmap_features.bin";

    MapStore store;
    auto t0 = std::chrono::steady_clock::now();
    if (!store.load(map_path, 4.0)) {
        std::cerr << "Failed to load map: " << map_path << "\n";
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Map loaded: " << store.size() << " scale " << store.scale()
              << " in "
              << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    if (store.loadCache(cache_path)) {
        std::cout << "Cache loaded: " << store.keypoints().size()
                  << " features\n";
    } else {
        std::cout << "Building SURF features (block-wise)...\n";
        store.buildFeatures(1024, 128, 800.0);
        auto t2 = std::chrono::steady_clock::now();
        std::cout << "Features: " << store.keypoints().size() << " in "
                  << std::chrono::duration<double>(t2 - t1).count() << "s\n";
        if (!store.saveCache(cache_path))
            std::cerr << "Cache save failed\n";
    }
    return 0;
}