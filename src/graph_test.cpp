#include "slam/graph_slam.h"
#include "slam/se3.h"
#include "slam/types.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>

using namespace slam;

// ============================================================
// Helper: generate ground truth trajectory (same as EKF test)
// ============================================================
struct GroundTruth {
    std::vector<cv::Mat> T_cw_gt;
    std::vector<cv::Mat> T_mw_gt;
    cv::Mat T_mc_gt;
};

GroundTruth generateGroundTruth(int N, double dt, double radius, double omega) {
    GroundTruth gt;
    cv::Mat t_mc = (cv::Mat_<double>(3, 1) << 0.2, 0.5, 0.0);
    cv::Mat R_mc = cv::Mat::eye(3, 3, CV_64F);
    gt.T_mc_gt = makeT(R_mc, t_mc);

    for (int i = 0; i < N; ++i) {
        double t = i * dt;
        double x = radius * std::cos(omega * t);
        double z = radius * std::sin(omega * t);
        double yaw = -omega * t;
        double cy = std::cos(yaw), sy = std::sin(yaw);
        cv::Mat R_mw = (cv::Mat_<double>(3, 3) <<
                        cy, 0, sy, 0, 1, 0, -sy, 0, cy);
        cv::Mat t_mw = (cv::Mat_<double>(3, 1) << x, 0.0, z);
        cv::Mat T_mw = makeT(R_mw, t_mw);
        cv::Mat T_cw = se3_compose(T_mw, se3_inv(gt.T_mc_gt));
        gt.T_mw_gt.push_back(T_mw);
        gt.T_cw_gt.push_back(T_cw);
    }
    return gt;
}

cv::Mat addNoiseSE3(const cv::Mat& T, double std_pos, double std_rot, std::mt19937& rng) {
    cv::Mat xi(6, 1, CV_64F);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 3; ++i) xi.at<double>(i) = dist(rng) * std_pos;
    for (int i = 3; i < 6; ++i) xi.at<double>(i) = dist(rng) * std_rot;
    return se3_compose(T, se3_exp(xi));
}

double computeATE(const std::vector<cv::Mat>& est, const std::vector<cv::Mat>& gt) {
    double sum = 0.0;
    int N = std::min((int)est.size(), (int)gt.size());
    for (int i = 0; i < N; ++i) {
        cv::Mat d = trans(est[i]) - trans(gt[i]);
        sum += d.dot(d);
    }
    return std::sqrt(sum / N);
}

double computeRotError(const cv::Mat& R_est, const cv::Mat& R_gt) {
    cv::Mat R_err = R_est * R_gt.t();
    cv::Mat phi = so3_log(R_err);
    return cv::norm(phi) * 180.0 / M_PI;
}

// ============================================================
//  Test: Batch Graph Optimization
// ============================================================
void testBatchOptimization(const GroundTruth& gt, int N, int map_interval,
                           double vo_pos_noise, double vo_rot_noise,
                           double map_pos_noise, double map_rot_noise,
                           std::mt19937& rng) {
    std::cout << "\n----------------------------------------------------\n";
    std::cout << "  Batch Graph Optimization\n";
    std::cout << "----------------------------------------------------\n";

    GraphSLAM graph;

    // --- Add nodes ---
    std::normal_distribution<double> nd(0.0, 1.0);

    cv::Mat init_noise(6, 1, CV_64F);
    for (int i = 0; i < 3; ++i) init_noise.at<double>(i) = nd(rng) * 0.05;
    for (int i = 3; i < 6; ++i) init_noise.at<double>(i) = nd(rng) * 0.02;
    cv::Mat T_cw_init = se3_compose(gt.T_cw_gt[0], se3_exp(init_noise));

    cv::Mat mc_noise(6, 1, CV_64F);
    for (int i = 0; i < 6; ++i) mc_noise.at<double>(i) = nd(rng) * 0.05;
    cv::Mat T_mc_init = se3_compose(gt.T_mc_gt, se3_exp(mc_noise));

    // Camera pose nodes with slightly perturbed initial guesses
    // (to break the local minimum of VO factors at initial guess)
    std::vector<int> pose_ids(N);
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            pose_ids[i] = graph.addNode(T_cw_init);
        } else {
            cv::Mat delta_true = se3_compose(se3_inv(gt.T_cw_gt[i-1]), gt.T_cw_gt[i]);
            cv::Mat delta_noisy = addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng);
            cv::Mat T_guess = se3_compose(graph.getNode(pose_ids[i-1]), delta_noisy);
            // Add small extra perturbation to break local minimum
            cv::Mat extra_noise(6, 1, CV_64F);
            for (int j = 0; j < 3; ++j) extra_noise.at<double>(j) = nd(rng) * 0.01;
            for (int j = 3; j < 6; ++j) extra_noise.at<double>(j) = nd(rng) * 0.003;
            T_guess = se3_compose(T_guess, se3_exp(extra_noise));
            pose_ids[i] = graph.addNode(T_guess);
        }
    }

    // Extrinsic node (shared)
    int ext_id = graph.addNode(T_mc_init);

    // --- Add factors ---
    cv::Mat prior_info = cv::Mat::eye(6, 6, CV_64F) * 1e4;
    graph.addPriorFactor(pose_ids[0], gt.T_cw_gt[0], prior_info);

    // Weak prior on extrinsic (prevents drift in unobservable Y-translation)
    cv::Mat ext_prior_info = cv::Mat::eye(6, 6, CV_64F) * 1e-2;
    ext_prior_info.at<double>(1,1) = 1e-6;  // Y-translation is very weak
    graph.addPriorFactor(ext_id, T_mc_init, ext_prior_info);

    // VO factors
    cv::Mat vo_info = cv::Mat::eye(6, 6, CV_64F);
    vo_info.at<double>(0,0) = 1.0 / (vo_pos_noise * vo_pos_noise);
    vo_info.at<double>(1,1) = 1.0 / (vo_pos_noise * vo_pos_noise);
    vo_info.at<double>(2,2) = 1.0 / (vo_pos_noise * vo_pos_noise);
    vo_info.at<double>(3,3) = 1.0 / (vo_rot_noise * vo_rot_noise);
    vo_info.at<double>(4,4) = 1.0 / (vo_rot_noise * vo_rot_noise);
    vo_info.at<double>(5,5) = 1.0 / (vo_rot_noise * vo_rot_noise);

    for (int i = 1; i < N; ++i) {
        cv::Mat delta_true = se3_compose(se3_inv(gt.T_cw_gt[i-1]), gt.T_cw_gt[i]);
        cv::Mat delta_noisy = addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng);
        graph.addVOFactor(pose_ids[i-1], pose_ids[i], delta_noisy, vo_info);
    }

    // Minimap factors
    cv::Mat map_info = cv::Mat::eye(3, 3, CV_64F);
    map_info.at<double>(0,0) = 1.0 / (map_pos_noise * map_pos_noise);
    map_info.at<double>(1,1) = 1.0 / (map_pos_noise * map_pos_noise);
    map_info.at<double>(2,2) = 1.0 / (map_rot_noise * map_rot_noise);

    for (int i = 0; i < N; ++i) {
        if (i % map_interval == 0) {
            cv::Mat T_mw = gt.T_mw_gt[i];
            cv::Mat t = trans(T_mw);
            cv::Mat R = rot(T_mw);
            double f_x = R.at<double>(0, 2);
            double f_z = R.at<double>(2, 2);
            double theta = std::atan2(f_x, f_z);

            cv::Mat obs(3, 1, CV_64F);
            obs.at<double>(0) = t.at<double>(0) + nd(rng) * map_pos_noise;
            obs.at<double>(1) = t.at<double>(2) + nd(rng) * map_pos_noise;
            obs.at<double>(2) = theta + nd(rng) * map_rot_noise;

            graph.addMapFactor(pose_ids[i], ext_id, obs, map_info);
        }
    }

    // --- Optimize ---
    std::cout << "Initial chi2: " << graph.getChi2() << "\n";
    graph.optimize(30, 1.0);
    std::cout << "Final chi2:   " << graph.getChi2() << "\n";

    // --- Extract results ---
    std::vector<cv::Mat> est_poses(N);
    for (int i = 0; i < N; ++i) {
        est_poses[i] = graph.getNode(pose_ids[i]);
    }
    cv::Mat T_mc_est = graph.getNode(ext_id);

    // --- Compute character pose errors ---
    std::vector<cv::Mat> est_char_poses(N);
    for (int i = 0; i < N; ++i) {
        est_char_poses[i] = se3_compose(est_poses[i], T_mc_est);
    }
    double char_ate = computeATE(est_char_poses, gt.T_mw_gt);

    // --- Compute errors ---
    double ate = computeATE(est_poses, gt.T_cw_gt);
    cv::Mat t_est = trans(T_mc_est);
    cv::Mat t_gt = trans(gt.T_mc_gt);
    double ext_pos_err = cv::norm(t_est - t_gt);
    double ext_rot_err = computeRotError(rot(T_mc_est), rot(gt.T_mc_gt));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Camera ATE:        " << ate << " m\n";
    std::cout << "Character ATE:     " << char_ate << " m\n";
    std::cout << "Extrinsic t error: " << ext_pos_err << " m\n";
    std::cout << "Extrinsic R error: " << ext_rot_err << " deg\n";
    std::cout << "Est T_mc t: [" << t_est.at<double>(0) << ", "
              << t_est.at<double>(1) << ", " << t_est.at<double>(2) << "]^T\n";
    std::cout << "True T_mc t: [" << t_gt.at<double>(0) << ", "
              << t_gt.at<double>(1) << ", " << t_gt.at<double>(2) << "]^T\n";

    // Per-frame errors
    std::cout << "\nPer-frame camera position errors:\n";
    for (int i = 0; i < N; i += 5) {
        cv::Mat d = trans(est_poses[i]) - trans(gt.T_cw_gt[i]);
        std::cout << "  frame " << i << ": " << cv::norm(d) << " m\n";
    }
}

// ============================================================
//  Main
// ============================================================
int main() {
    std::cout << "====================================================\n";
    std::cout << "  Genshin SLAM - Graph Optimization Test\n";
    std::cout << "====================================================\n";

    const double radius = 5.0;
    const double omega = 0.3;
    const double dt = 0.1;
    const int N = 40;
    const int map_interval = 5;

    const double vo_pos_noise = 0.02;
    const double vo_rot_noise = 0.005;
    const double map_pos_noise = 0.15;
    const double map_rot_noise = 0.05;

    std::mt19937 rng(42);

    std::cout << "\nTrajectory: circle r=" << radius << "m, "
              << N << " frames, minimap every " << map_interval << " frames\n";
    GroundTruth gt = generateGroundTruth(N, dt, radius, omega);
    std::cout << "True T_mc: t = [0.2, 0.5, 0.0]^T\n";

    testBatchOptimization(gt, N, map_interval,
                          vo_pos_noise, vo_rot_noise,
                          map_pos_noise, map_rot_noise, rng);

    std::cout << "\n====================================================\n";
    std::cout << "  Done.\n";
    std::cout << "====================================================\n";
    return 0;
}