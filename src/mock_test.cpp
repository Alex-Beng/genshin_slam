#include "slam/ekf_slam.h"
#include "slam/se3.h"
#include "slam/types.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>

using namespace slam;

// ============================================================
// Mock test: circular trajectory + noisy VO + noisy minimap
// ============================================================

struct GroundTruth {
    std::vector<cv::Mat> T_cw_gt;   // camera poses (world frame)
    std::vector<cv::Mat> T_mw_gt;   // character poses (world frame)
    cv::Mat T_mc_gt;                // true extrinsic
};

// Generate ground truth trajectory
GroundTruth generateGroundTruth(int N, double dt, double radius, double omega) {
    GroundTruth gt;
    // True extrinsic: camera is above and slightly in front of character
    // Character->Camera: translate (0.2, 0.5, 0.0) in character frame
    cv::Mat t_mc = (cv::Mat_<double>(3, 1) << 0.2, 0.5, 0.0);
    cv::Mat R_mc = cv::Mat::eye(3, 3, CV_64F);  // identity rotation
    gt.T_mc_gt = makeT(R_mc, t_mc);

    for (int i = 0; i < N; ++i) {
        double t = i * dt;
        double x = radius * std::cos(omega * t);
        double z = radius * std::sin(omega * t);

        // Character yaw: faces tangent to circle
        // Forward direction: [-sin(ωt), 0, cos(ωt)]^T
        double yaw = -omega * t;
        cv::Mat R_mw;
        double cy = std::cos(yaw);
        double sy = std::sin(yaw);
        // Ry(yaw) = [cy, 0, sy; 0, 1, 0; -sy, 0, cy]
        R_mw = (cv::Mat_<double>(3, 3) <<
                cy, 0, sy,
                0,  1, 0,
                -sy, 0, cy);

        cv::Mat t_mw = (cv::Mat_<double>(3, 1) << x, 0.0, z);
        cv::Mat T_mw = makeT(R_mw, t_mw);

        // Camera pose: T_cw = T_mw * T_mc^{-1}
        cv::Mat T_mc_inv = se3_inv(gt.T_mc_gt);
        cv::Mat T_cw = se3_compose(T_mw, T_mc_inv);

        gt.T_mw_gt.push_back(T_mw);
        gt.T_cw_gt.push_back(T_cw);
    }
    return gt;
}

// Add Gaussian noise to a vector
cv::Mat addNoise(const cv::Mat& v, double std_pos, double std_rot, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    cv::Mat noisy = v.clone();
    for (int i = 0; i < 3; ++i) {
        noisy.at<double>(i) += dist(rng) * std_pos;
    }
    for (int i = 3; i < 6; ++i) {
        noisy.at<double>(i) += dist(rng) * std_rot;
    }
    return noisy;
}

// Add noise to SE(3) transformation
cv::Mat addNoiseSE3(const cv::Mat& T, double std_pos, double std_rot, std::mt19937& rng) {
    cv::Mat xi(6, 1, CV_64F);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 3; ++i) xi.at<double>(i) = dist(rng) * std_pos;
    for (int i = 3; i < 6; ++i) xi.at<double>(i) = dist(rng) * std_rot;
    return se3_compose(T, se3_exp(xi));
}

// Compute ATE between estimated and ground truth trajectory
double computeATE(const std::vector<cv::Mat>& est, const std::vector<cv::Mat>& gt) {
    double sum = 0.0;
    int N = std::min(est.size(), gt.size());
    for (int i = 0; i < N; ++i) {
        cv::Mat t_est = trans(est[i]);
        cv::Mat t_gt  = trans(gt[i]);
        cv::Mat diff = t_est - t_gt;
        sum += diff.dot(diff);
    }
    return std::sqrt(sum / N);
}

// Compute rotation error (degrees)
double computeRotError(const cv::Mat& R_est, const cv::Mat& R_gt) {
    cv::Mat R_err = R_est * R_gt.t();
    cv::Mat phi = so3_log(R_err);
    return cv::norm(phi) * 180.0 / M_PI;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Genshin SLAM - EKF Mock Test\n";
    std::cout << "====================================================\n\n";

    // --- Parameters ---
    const double radius = 5.0;
    const double omega = 0.3;
    const double dt = 0.1;
    const int N = 200;  // 20 seconds at 10 Hz
    const int map_interval = 10;  // minimap every 1 sec

    // Noise levels
    const double vo_pos_noise = 0.02;    // m
    const double vo_rot_noise = 0.005;   // rad
    const double map_pos_noise = 0.15;   // m
    const double map_rot_noise = 0.05;   // rad

    // Initial uncertainty
    const double init_pos_unc = 0.1;     // m
    const double init_rot_unc = 0.05;    // rad

    std::mt19937 rng(42);

    // --- Generate ground truth ---
    std::cout << "Generating ground truth trajectory (circle, r="
              << radius << "m, " << N << " frames)...\n";
    GroundTruth gt = generateGroundTruth(N, dt, radius, omega);

    std::cout << "True extrinsic T_mc: t = ["
              << gt.T_mc_gt.at<double>(0) << ", "
              << gt.T_mc_gt.at<double>(1) << ", "
              << gt.T_mc_gt.at<double>(2) << "]^T\n\n";

    // --- Initialize EKF with perturbed ground truth ---
    std::cout << "Initializing EKF...\n";
    slam::EKFSLAM ekf;

    // Perturb initial T_cw
    cv::Mat init_noise(6, 1, CV_64F);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 3; ++i) init_noise.at<double>(i) = dist(rng) * 0.05;
    for (int i = 3; i < 6; ++i) init_noise.at<double>(i) = dist(rng) * 0.02;
    cv::Mat T_cw_init = se3_compose(gt.T_cw_gt[0], se3_exp(init_noise));

    // Perturb initial T_mc (wrong extrinsic)
    cv::Mat mc_noise(6, 1, CV_64F);
    mc_noise.at<double>(0) = dist(rng) * 0.1;
    mc_noise.at<double>(1) = dist(rng) * 0.1;
    mc_noise.at<double>(2) = dist(rng) * 0.1;
    mc_noise.at<double>(3) = dist(rng) * 0.05;
    mc_noise.at<double>(4) = dist(rng) * 0.05;
    mc_noise.at<double>(5) = dist(rng) * 0.05;
    cv::Mat T_mc_init = se3_compose(gt.T_mc_gt, se3_exp(mc_noise));

    // Initial covariance
    cv::Mat P_init = cv::Mat::zeros(12, 12, CV_64F);
    for (int i = 0; i < 3; ++i) P_init.at<double>(i, i) = init_pos_unc * init_pos_unc;
    for (int i = 3; i < 6; ++i) P_init.at<double>(i, i) = init_rot_unc * init_rot_unc;
    for (int i = 6; i < 9; ++i) P_init.at<double>(i, i) = init_pos_unc * init_pos_unc;
    for (int i = 9; i < 12; ++i) P_init.at<double>(i, i) = init_rot_unc * init_rot_unc;

    ekf.init(T_cw_init, T_mc_init, P_init);

    // Process noise Q (small random walk on T_mc)
    cv::Mat Q = cv::Mat::zeros(12, 12, CV_64F);
    double q_pos = 1e-6;
    double q_rot = 1e-6;
    for (int i = 0; i < 3; ++i) Q.at<double>(i, i) = q_pos;
    for (int i = 3; i < 6; ++i) Q.at<double>(i, i) = q_rot;
    for (int i = 6; i < 9; ++i) Q.at<double>(i, i) = q_pos;
    for (int i = 9; i < 12; ++i) Q.at<double>(i, i) = q_rot;

    // Observation noise R
    cv::Mat R_map = cv::Mat::zeros(3, 3, CV_64F);
    R_map.at<double>(0, 0) = map_pos_noise * map_pos_noise;
    R_map.at<double>(1, 1) = map_pos_noise * map_pos_noise;
    R_map.at<double>(2, 2) = map_rot_noise * map_rot_noise;

    // --- Run EKF ---
    std::vector<cv::Mat> estimated_poses;
    std::vector<cv::Mat> estimated_extrinsics;
    std::vector<double> nees_history;  // normalized estimation error squared

    std::cout << "Running EKF simulation...\n";

    for (int i = 0; i < N; ++i) {
        // --- VO measurement (with noise) ---
        cv::Mat delta_T;
        if (i > 0) {
            cv::Mat T_prev = gt.T_cw_gt[i-1];
            cv::Mat T_curr = gt.T_cw_gt[i];
            // True relative motion: delta = T_prev^{-1} * T_curr
            cv::Mat delta_true = se3_compose(se3_inv(T_prev), T_curr);
            // Add noise
            delta_T = addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng);
        } else {
            delta_T = cv::Mat::eye(4, 4, CV_64F);
        }

        // --- EKF predict ---
        ekf.predict(delta_T, Q);

        // --- Minimap observation (low frequency) ---
        if (i % map_interval == 0) {
            // Ground truth character pose
            cv::Mat T_mw = gt.T_mw_gt[i];

            // Extract observation and add noise
            cv::Mat t = trans(T_mw);
            cv::Mat R = rot(T_mw);
            double f_x = R.at<double>(0, 2);
            double f_z = R.at<double>(2, 2);
            double theta = std::atan2(f_x, f_z);

            cv::Mat obs(3, 1, CV_64F);
            std::normal_distribution<double> nd(0.0, 1.0);
            obs.at<double>(0) = t.at<double>(0) + nd(rng) * map_pos_noise;
            obs.at<double>(1) = t.at<double>(2) + nd(rng) * map_pos_noise;
            obs.at<double>(2) = theta + nd(rng) * map_rot_noise;

            // --- EKF update ---
            ekf.update(obs, R_map);
        }

        // Store estimate
        estimated_poses.push_back(ekf.getCameraPose());
        estimated_extrinsics.push_back(ekf.getExtrinsic());
    }

    // --- Results ---
    std::cout << "\n====================================================\n";
    std::cout << "  Results\n";
    std::cout << "====================================================\n\n";

    // Camera pose ATE
    double ate = computeATE(estimated_poses, gt.T_cw_gt);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Camera ATE (position): " << ate << " m\n";

    // Character pose ATE (more relevant for navigation)
    std::vector<cv::Mat> est_char_poses;
    for (const auto& T_cw : estimated_poses) {
        est_char_poses.push_back(se3_compose(T_cw, ekf.getExtrinsic()));
    }
    double char_ate = computeATE(est_char_poses, gt.T_mw_gt);
    std::cout << "Character ATE (position): " << char_ate << " m\n";

    // Extrinsic calibration error
    cv::Mat T_mc_est = estimated_extrinsics.back();
    cv::Mat t_est = trans(T_mc_est);
    cv::Mat t_gt = trans(gt.T_mc_gt);
    cv::Mat t_err = t_est - t_gt;
    double pos_err = cv::norm(t_err);
    double rot_err = computeRotError(rot(T_mc_est), rot(gt.T_mc_gt));
    std::cout << "Extrinsic translation error: " << pos_err << " m\n";
    std::cout << "Extrinsic rotation error: " << rot_err << " deg\n";

    std::cout << "\nEstimated extrinsic T_mc: t = ["
              << t_est.at<double>(0) << ", "
              << t_est.at<double>(1) << ", "
              << t_est.at<double>(2) << "]^T\n";
    std::cout << "True  extrinsic T_mc: t = ["
              << t_gt.at<double>(0) << ", "
              << t_gt.at<double>(1) << ", "
              << t_gt.at<double>(2) << "]^T\n";

    // Per-frame errors (first 10 and last 10)
    std::cout << "\n--- Per-frame position errors (first 10) ---\n";
    for (int i = 0; i < std::min(10, N); ++i) {
        cv::Mat t_e = trans(estimated_poses[i]);
        cv::Mat t_g = trans(gt.T_cw_gt[i]);
        cv::Mat diff = t_e - t_g;
        double err = cv::norm(diff);
        std::cout << "  frame " << i << ": err = " << err << " m\n";
    }

    std::cout << "\n--- Per-frame position errors (last 10) ---\n";
    for (int i = N - 10; i < N; ++i) {
        cv::Mat t_e = trans(estimated_poses[i]);
        cv::Mat t_g = trans(gt.T_cw_gt[i]);
        cv::Mat diff = t_e - t_g;
        double err = cv::norm(diff);
        std::cout << "  frame " << i << ": err = " << err << " m\n";
    }

    std::cout << "\n====================================================\n";
    std::cout << "  Test complete.\n";
    std::cout << "====================================================\n";

    return 0;
}