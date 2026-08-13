#include "slam/capture.h"
#include "slam/minimap_pipeline.h"
#include <iostream>
#include <iomanip>

using namespace slam;

int main(int argc, char** argv) {
    const std::string video =
        argc > 1 ? argv[1]
                 : "F:/Backup/录屏/原神 2022-01-03 11-14-53.mp4";

    VideoFileCapture cap(video);
    if (!cap.init()) {
        std::cerr << "Could not open " << video << "\n";
        return 1;
    }
    std::cout << "Video: " << cap.frameCount() << " frames @ "
              << cap.fps() << " fps\n";

    MinimapPipeline mm;

    cv::Mat frame, minimap;
    cv::Rect rect;
    int count = 0;
    double t = cv::getTickCount();
    while (cap.read(frame)) {
        if (!mm.extract(frame, minimap, rect)) continue;
        ++count;

        // draw the minimap rect on a copy for visualization
        cv::Mat viz = frame.clone();
        cv::rectangle(viz, rect, cv::Scalar(0, 255, 0), 2);

        if (count % 30 == 1 || count == cap.frameCount()) {
            std::cout << "frame " << count << " minimap rect "
                      << rect << "\n";
        }
        if (count == 3) {
            // save a couple frames for inspection
            cv::imwrite("capture_minimap.png", minimap);
            cv::imwrite("capture_rect.png", viz);
        }
        if (count > 60) break;
    }
    double ms = 1000.0 * (cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Processed " << count << " frames in " << std::fixed
              << std::setprecision(1) << ms << " ms total\n";

    cap.release();
    return 0;
}