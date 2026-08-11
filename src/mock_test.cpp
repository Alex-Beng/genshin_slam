#include "slam/ekf_slam.h"
#include "slam/types.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using namespace slam;

// ============================================================
// Mock test: circular trajectory + noisy VO + noisy minimap
// ============================================================

struct GroundTruth {
    std::vector<SE3> T_cw_gt;  // camera poses
    std::vector<SE3> T_mw_gt;  // character poses
    SE3 T_mc_gt;               // true extrinsic
};

// Y-UP rotation about the Y axis by `yaw`.
SE3 rotY(double yaw) {
    Eigen::AngleAxisd aa(yaw, Eigen::Vector3d::UnitY());
    return SE3(SO3(Eigen::Matrix3d(aa)), Eigen::Vector3d::Zero());
}

GroundTruth generateGroundTruth(int N, double dt, double radius, double omega) {
    GroundTruth gt;
    gt.T_mc_gt = makeT(Matrix3::Identity(), Eigen::Vector3d(0.2, 0.5, 0.0));

    for (int i = 0; i < N; ++i) {
        const double t = i * dt;
        const double x = radius * std::cos(omega * t);
        const double z = radius * std::sin(omega * t);
        const double yaw = -omega * t;  // face tangent to circle

        SE3 R_mw = rotY(yaw);
        SE3 T_mw(R_mw.rotationMatrix(), Eigen::Vector3d(x, 0.0, z));

        SE3 T_cw = T_mw * gt.T_mc_gt.inverse();
        gt.T_mw_gt.push_back(T_mw);
        gt.T_cw_gt.push_back(T_cw);
    }
    return gt;
}

SE3 addNoiseSE3(const SE3& T, double std_pos, double std_rot, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Vector6 xi = Vector6::Zero();
    for (int i = 0; i < 3; ++i) xi(i) = dist(rng) * std_pos;
    for (int i = 3; i < 6; ++i) xi(i) = dist(rng) * std_rot;
    return T * SE3::exp(xi);
}

double computeATE(const std::vector<SE3>& est, const std::vector<SE3>& gt) {
    double sum = 0.0;
    const int n = (int)std::min(est.size(), gt.size());
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3d d = est[i].translation() - gt[i].translation();
        sum += d.dot(d);
    }
    return std::sqrt(sum / n);
}

double computeRotError(const SO3& R_est, const SO3& R_gt) {
    const SO3 R_err = R_est * R_gt.inverse();
    return R_err.log().norm() * 180.0 / M_PI;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Genshin SLAM - EKF Mock Test (Sophus + Eigen)\n";
    std::cout << "====================================================\n\n";

    const double radius = 5.0;
    const double omega = 0.3;
    const double dt = 0.1;
    const int N = 200;             // 20 s @ 10 Hz
    const int map_interval = 10;   // minimap every 1 s

    const double vo_pos_noise = 0.02;
    const double vo_rot_noise = 0.005;
    const double map_pos_noise = 0.15;
    const double map_rot_noise = 0.05;

    const double odom_pos_noise = 0.02;   // m per 0.1s step
    const double odom_rot_noise = 0.01;   // rad per 0.1s step

    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::cout << "Generating ground truth (circle r=" << radius << "m, "
              << N << " frames)...\n";
    GroundTruth gt = generateGroundTruth(N, dt, radius, omega);

    std::cout << "True T_mc: t = ["
              << gt.T_mc_gt.translation().transpose() << "]^T\n\n";

    // --- Initialize EKF with perturbed state ---
    Vector6 init_noise = Vector6::Zero();
    for (int i = 0; i < 3; ++i) init_noise(i) = nd(rng) * 0.05;
    for (int i = 3; i < 6; ++i) init_noise(i) = nd(rng) * 0.02;
    SE3 T_cw_init = gt.T_cw_gt[0] * SE3::exp(init_noise);

    Vector6 mc_noise = Vector6::Zero();
    for (int i = 0; i < 6; ++i) mc_noise(i) = nd(rng) * 0.05;
    SE3 T_mc_init = gt.T_mc_gt * SE3::exp(mc_noise);

    Matrix12 P_init = Matrix12::Zero();
    for (int i = 0; i < 3; ++i) P_init(i, i) = 0.1 * 0.1;
    for (int i = 3; i < 6; ++i) P_init(i, i) = 0.05 * 0.05;
    for (int i = 6; i < 9; ++i) P_init(i, i) = 0.1 * 0.1;
    for (int i = 9; i < 12; ++i) P_init(i, i) = 0.05 * 0.05;

    EKFSLAM ekf;
    ekf.init(T_cw_init, T_mc_init, P_init);

    // Process noise: VO measurement noise enters T_cw block per step.
    Matrix12 Q = Matrix12::Zero();
    for (int i = 0; i < 3; ++i) Q(i, i) = vo_pos_noise * vo_pos_noise;
    for (int i = 3; i < 6; ++i) Q(i, i) = vo_rot_noise * vo_rot_noise;
    for (int i = 6; i < 9; ++i) Q(i, i) = 1e-6;   // T_mc pos random walk
    for (int i = 9; i < 12; ++i) Q(i, i) = 1e-6;   // T_mc rot random walk

    // Observation noise
    Matrix3 R_map = Matrix3::Zero();
    R_map(0, 0) = map_pos_noise * map_pos_noise;
    R_map(1, 1) = map_pos_noise * map_pos_noise;
    R_map(2, 2) = map_rot_noise * map_rot_noise;

    // Minimap SE(2) odometry noise
    Matrix3 R_odom = Matrix3::Zero();
    R_odom(0, 0) = odom_pos_noise * odom_pos_noise;
    R_odom(1, 1) = odom_pos_noise * odom_pos_noise;
    R_odom(2, 2) = odom_rot_noise * odom_rot_noise;

    std::vector<SE3> estimated_poses;
    estimated_poses.reserve(N);

    std::cout << "Running EKF simulation...\n";
    for (int i = 0; i < N; ++i) {
        // VO measurement (with noise)
        SE3 delta_T = SE3();
        if (i > 0) {
            const SE3 delta_true = gt.T_cw_gt[i - 1].inverse() * gt.T_cw_gt[i];
            delta_T = addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng);
        }
        ekf.predict(delta_T, Q);

        // Minimap frame-to-frame odometry (world-frame displacement)
        if (i > 0) {
            const Eigen::Vector3d dx_prev = gt.T_mw_gt[i - 1].translation();
            const Eigen::Vector3d dx_cur = gt.T_mw_gt[i].translation();
            const Matrix3 R_prev = gt.T_mw_gt[i - 1].rotationMatrix();
            const Matrix3 R_cur = gt.T_mw_gt[i].rotationMatrix();
            Eigen::Vector3d d_true;
            d_true << dx_cur(0) - dx_prev(0),
                      dx_cur(2) - dx_prev(2),
                      std::atan2(R_cur(0, 2), R_cur(2, 2)) - std::atan2(R_prev(0, 2), R_prev(2, 2));
            while (d_true(2) > M_PI) d_true(2) -= 2.0 * M_PI;
            while (d_true(2) < -M_PI) d_true(2) += 2.0 * M_PI;
            Eigen::Vector3d d_meas;
            d_meas << d_true(0) + nd(rng) * odom_pos_noise,
                      d_true(1) + nd(rng) * odom_pos_noise,
                      d_true(2) + nd(rng) * odom_rot_noise;
            ekf.updateMinimapOdom(d_meas, R_odom);
        }

        // Minimap observation
        if (i % map_interval == 0) {
            const SE3& Tmw = gt.T_mw_gt[i];
            const Eigen::Vector3d t = Tmw.translation();
            const Eigen::Matrix3d R = Tmw.rotationMatrix();
            MapObs obs;
            obs << t(0) + nd(rng) * map_pos_noise,
                   t(2) + nd(rng) * map_pos_noise,
                   std::atan2(R(0, 2), R(2, 2)) + nd(rng) * map_rot_noise;
            ekf.update(obs, R_map);
        }
        estimated_poses.push_back(ekf.getCameraPose());
    }

    // --- Results ---
    std::cout << "\n====================================================\n";
    std::cout << "  Results\n";
    std::cout << "====================================================\n\n";
    std::cout << std::fixed << std::setprecision(4);

    const double ate = computeATE(estimated_poses, gt.T_cw_gt);
    std::cout << "Camera ATE (position):     " << ate << " m\n";

    std::vector<SE3> est_char_poses;
    for (const auto& T_cw : estimated_poses)
        est_char_poses.push_back(T_cw * ekf.getExtrinsic());
    const double char_ate = computeATE(est_char_poses, gt.T_mw_gt);
    std::cout << "Character ATE (position):  " << char_ate << " m\n";

    const SE3& T_mc_est = ekf.getExtrinsic();
    const double ext_pos_err =
        (T_mc_est.translation() - gt.T_mc_gt.translation()).norm();
    const double ext_rot_err = computeRotError(T_mc_est.rotationMatrix(),
                                               gt.T_mc_gt.rotationMatrix());
    std::cout << "Extrinsic translation err: " << ext_pos_err << " m\n";
    std::cout << "Extrinsic rotation err:    " << ext_rot_err << " deg\n";
    std::cout << "Est T_mc t: [" << T_mc_est.translation().transpose() << "]^T\n";
    std::cout << "True T_mc t: [" << gt.T_mc_gt.translation().transpose() << "]^T\n";

    std::cout << "\nPer-frame camera position errors:\n";
    const auto print = [&](int a, int b) {
        for (int i = a; i < b; ++i) {
            const double e = (estimated_poses[i].translation() -
                              gt.T_cw_gt[i].translation()).norm();
            std::cout << "  frame " << i << ": " << e << " m\n";
        }
    };
    print(0, 20);
    for (int qi = 20; qi < N; qi += 20) {
        const double e = (estimated_poses[qi].translation() -
                          gt.T_cw_gt[qi].translation()).norm();
        std::cout << "  frame " << qi << ": " << e << " m\n";
    }

    std::cout << "\n====================================================\n";
    std::cout << "  Test complete.\n";
    std::cout << "====================================================\n";
    return 0;
}












