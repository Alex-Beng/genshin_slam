#include "slam/vo_frontend.h"
#include "slam/ekf_slam.h"
#include <iostream>
#include <iomanip>

using namespace slam;

int main(int argc, char** argv) {
    const char* video_path =
        argc > 1 ? argv[1] : "F:/Backup/录屏/genshin_cali.mp4";

    std::cout << "Opening video: " << video_path << "\n";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: could not open " << video_path << "\n";
        return 1;
    }

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "  " << w << "x" << h << "  " << fps << " fps  "
              << total << " frames\n";

    // Default camera intrinsics (to be replaced by calibration)
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 1000.0;  K(1, 1) = 1000.0;
    K(0, 2) = w / 2.0; K(1, 2) = h / 2.0;

    VOFrontend vo(K, 2000);

    // EKF state: identity initial pose, identity extrinsic (no minimap here)
    EKFSLAM ekf;
    ekf.init(SE3(), SE3(), Matrix12::Identity() * 0.1);

    // Process noise: large for pure VO (no scale, no absolute constraints)
    Matrix12 Q = Matrix12::Zero();
    for (int i = 0; i < 3; ++i) Q(i, i) = 0.1;
    for (int i = 3; i < 6; ++i) Q(i, i) = 0.01;
    for (int i = 6; i < 12; ++i) Q(i, i) = 1e-8;

    cv::Mat frame;
    int frame_idx = 0;
    SE3 trajectory;  // accumulated camera pose (no scale correction)

    std::cout << "\nProcessing frames...\n";
    std::cout << "frame  inliers  |dt|      camera_x    camera_z  yaw\n";

    while (cap.read(frame)) {
        SE3 delta = vo.processFrame(frame);
        int inliers = vo.inlierCount();

        // Accumulate unscaled trajectory for display
        trajectory = trajectory * delta;

        // Feed to EKF (drives the estimate)
        ekf.predict(delta, Q);
        SE3 cam = ekf.getCameraPose();

        // Extract yaw from camera rotation
        Eigen::Matrix3d R = cam.rotationMatrix();
        double yaw = std::atan2(R(0, 2), R(2, 2));

        if (frame_idx < 5 || frame_idx % 50 == 0) {
            std::cout << std::setw(5) << frame_idx << "  "
                      << std::setw(4) << inliers << "     "
                      << std::fixed << std::setprecision(3)
                      << delta.log().norm() << "     "
                      << cam.translation()(0) << "  "
                      << cam.translation()(2) << "  "
                      << yaw << "\n";
        }

        ++frame_idx;
        if (frame_idx >= total) break;
    }

    std::cout << "\nProcessed " << frame_idx << " frames.\n";
    cap.release();
    return 0;
}