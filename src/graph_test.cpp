#include "slam/graph_slam.h"
#include "slam/types.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using namespace slam;

// ============================================================
// Mock test: batch graph optimization with Ceres
// ============================================================

struct GroundTruth {
    std::vector<SE3> T_cw_gt;
    std::vector<SE3> T_mw_gt;
    SE3 T_mc_gt;
};

SE3 rotY(double yaw) {
    Eigen::AngleAxisd aa(yaw, Eigen::Vector3d::UnitY());
    return SE3(SO3(Eigen::Matrix3d(aa)), Eigen::Vector3d::Zero());
}

GroundTruth generateGroundTruth(int N, double dt, double radius, double omega) {
    GroundTruth gt;
    gt.T_mc_gt = makeT(Matrix3::Identity(), Eigen::Vector3d(0.2, 0.5, 0.0));
    for (int i = 0; i < N; ++i) {
        const double t = i * dt;
        const double yaw = -omega * t;
        SE3 T_mw(rotY(yaw).rotationMatrix(),
                 Eigen::Vector3d(radius * std::cos(omega * t), 0.0,
                                 radius * std::sin(omega * t)));
        gt.T_mw_gt.push_back(T_mw);
        gt.T_cw_gt.push_back(T_mw * gt.T_mc_gt.inverse());
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
    return (R_est * R_gt.inverse()).log().norm() * 180.0 / M_PI;
}

void infoToInfo6(Matrix6& info, const double p0, const double p1, const double p2,
                 const double r0, const double r1, const double r2) {
    info = Matrix6::Zero();
    info(0, 0) = p0; info(1, 1) = p1; info(2, 2) = p2;
    info(3, 3) = r0; info(4, 4) = r1; info(5, 5) = r2;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Genshin SLAM - Graph Optimization Test (Ceres)\n";
    std::cout << "====================================================\n";

    const double radius = 5.0;
    const double omega = 0.3;
    const double dt = 0.1;
    const int N = 200;             // 20 s
    const int map_interval = 10;   // minimap every 1 s

    const double vo_pos_noise = 0.02;
    const double vo_rot_noise = 0.005;
    const double map_pos_noise = 0.15;
    const double map_rot_noise = 0.05;

    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::cout << "Trajectory: circle r=" << radius << "m, " << N
              << " frames, minimap every " << map_interval << " frames\n";
    GroundTruth gt = generateGroundTruth(N, dt, radius, omega);
    std::cout << "True T_mc: t = [" << gt.T_mc_gt.translation().transpose()
              << "]^T\n\n";

    GraphSLAM graph;

    // --- Nodes: perturbed initial guesses ---
    Vector6 init_noise = Vector6::Zero();
    for (int i = 0; i < 3; ++i) init_noise(i) = nd(rng) * 0.05;
    for (int i = 3; i < 6; ++i) init_noise(i) = nd(rng) * 0.02;
    SE3 T_cw0_init = gt.T_cw_gt[0] * SE3::exp(init_noise);

    Vector6 mc_noise = Vector6::Zero();
    for (int i = 0; i < 6; ++i) mc_noise(i) = nd(rng) * 0.05;
    SE3 T_mc_init = gt.T_mc_gt * SE3::exp(mc_noise);

    std::vector<int> pose_ids(N);
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            pose_ids[i] = graph.addNode(T_cw0_init);
        } else {
            const SE3 delta_true = gt.T_cw_gt[i - 1].inverse() * gt.T_cw_gt[i];
            SE3 T_guess = graph.getNode(pose_ids[i - 1]) *
                          addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng);
            // Small extra jitter to break exact VO consistency
            Vector6 jitter = Vector6::Zero();
            jitter(0) = nd(rng) * 0.01; jitter(1) = nd(rng) * 0.01;
            jitter(2) = nd(rng) * 0.01;
            jitter(3) = nd(rng) * 0.003; jitter(4) = nd(rng) * 0.003;
            jitter(5) = nd(rng) * 0.003;
            T_guess = T_guess * SE3::exp(jitter);
            pose_ids[i] = graph.addNode(T_guess);
        }
    }
    int ext_id = graph.addNode(T_mc_init);

    // --- Factors ---
    Matrix6 pri_info = Matrix6::Identity() * 1e4;
    graph.addPriorFactor(pose_ids[0], gt.T_cw_gt[0], pri_info);

// Prior on extrinsic: approximate calibration known at boot
    // (moderate weight; matches the EKF test's T_mc init uncertainty)
    Matrix6 ext_pri_info = Matrix6::Zero();
    for (int i = 0; i < 3; ++i) ext_pri_info(i, i) = 1.0 / (0.1 * 0.1);
    for (int i = 3; i < 6; ++i) ext_pri_info(i, i) = 1.0 / (0.05 * 0.05);
    graph.addPriorFactor(ext_id, T_mc_init, ext_pri_info);

    Matrix6 vo_info;
    infoToInfo6(vo_info, 1.0 / (vo_pos_noise * vo_pos_noise),
                1.0 / (vo_pos_noise * vo_pos_noise),
                1.0 / (vo_pos_noise * vo_pos_noise),
                1.0 / (vo_rot_noise * vo_rot_noise),
                1.0 / (vo_rot_noise * vo_rot_noise),
                1.0 / (vo_rot_noise * vo_rot_noise));
    for (int i = 1; i < N; ++i) {
        const SE3 delta_true = gt.T_cw_gt[i - 1].inverse() * gt.T_cw_gt[i];
        graph.addVOFactor(pose_ids[i - 1], pose_ids[i],
                          addNoiseSE3(delta_true, vo_pos_noise, vo_rot_noise, rng),
                          vo_info);
    }

    Matrix3 map_info = Matrix3::Zero();
    map_info(0, 0) = 1.0 / (map_pos_noise * map_pos_noise);
    map_info(1, 1) = 1.0 / (map_pos_noise * map_pos_noise);
    map_info(2, 2) = 1.0 / (map_rot_noise * map_rot_noise);
    for (int i = 0; i < N; ++i) {
        if (i % map_interval != 0) continue;
        const SE3& Tmw = gt.T_mw_gt[i];
        const Eigen::Matrix3d R = Tmw.rotationMatrix();
        MapObs obs;
        obs << Tmw.translation()(0) + nd(rng) * map_pos_noise,
               Tmw.translation()(2) + nd(rng) * map_pos_noise,
               std::atan2(R(0, 2), R(2, 2)) + nd(rng) * map_rot_noise;
graph.addMapFactor(pose_ids[i], ext_id, obs, map_info);
    }

    // Minimap frame-to-frame odometry factors (world-frame displacement)
    const double odom_pos_noise = 0.02;
    const double odom_rot_noise = 0.01;
    Matrix3 odom_info = Matrix3::Zero();
    odom_info(0, 0) = 1.0 / (odom_pos_noise * odom_pos_noise);
    odom_info(1, 1) = 1.0 / (odom_pos_noise * odom_pos_noise);
    odom_info(2, 2) = 1.0 / (odom_rot_noise * odom_rot_noise);
    for (int i = 1; i < N; ++i) {
        const Eigen::Vector3d ti = gt.T_mw_gt[i - 1].translation();
        const Eigen::Vector3d tj = gt.T_mw_gt[i].translation();
        const Eigen::Matrix3d Ri = gt.T_mw_gt[i - 1].rotationMatrix();
        const Eigen::Matrix3d Rj = gt.T_mw_gt[i].rotationMatrix();
        MapObs d_true;
        d_true << tj(0) - ti(0), tj(2) - ti(2),
                  std::atan2(Rj(0, 2), Rj(2, 2)) - std::atan2(Ri(0, 2), Ri(2, 2));
        while (d_true(2) > M_PI) d_true(2) -= 2.0 * M_PI;
        while (d_true(2) < -M_PI) d_true(2) += 2.0 * M_PI;
        MapObs d_meas;
        d_meas << d_true(0) + nd(rng) * odom_pos_noise,
                  d_true(1) + nd(rng) * odom_pos_noise,
                  d_true(2) + nd(rng) * odom_rot_noise;
        graph.addMinimapOdomFactor(pose_ids[i - 1], ext_id, pose_ids[i],
                                   d_meas, odom_info);
    }

    // --- Optimize ---
    std::cout << "Initial chi2: " << graph.getChi2() << "\n";
    {
        // debug: predicted vs measured odom delta for frame 1
        const SE3 Ti = graph.getNode(pose_ids[0]);
        const SE3 Tj = graph.getNode(pose_ids[1]);
        const SE3 Tmc = graph.getNode(ext_id);
        const SE3 mi = Ti * Tmc, mj = Tj * Tmc;
        Eigen::Vector3d pd;
        pd << mj.translation()(0) - mi.translation()(0),
              mj.translation()(2) - mi.translation()(2),
              std::atan2(mj.rotationMatrix()(0, 2), mj.rotationMatrix()(2, 2)) -
                  std::atan2(mi.rotationMatrix()(0, 2), mi.rotationMatrix()(2, 2));
        const Eigen::Vector3d td = gt.T_mw_gt[1].translation() - gt.T_mw_gt[0].translation();
        std::cout << "  dbg pred_delta=" << pd.transpose()
                  << " true_delta=" << td.transpose() << "\n";
    }
    graph.optimize(50, 1.0);
    std::cout << "Final chi2:   " << graph.getChi2() << "\n\n";

    // --- Results ---
    std::vector<SE3> est_poses(N);
    for (int i = 0; i < N; ++i) est_poses[i] = graph.getNode(pose_ids[i]);
    const SE3 T_mc_est = graph.getNode(ext_id);

    const double ate = computeATE(est_poses, gt.T_cw_gt);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Camera ATE:        " << ate << " m\n";

    std::vector<SE3> est_char(N);
    for (int i = 0; i < N; ++i) est_char[i] = est_poses[i] * T_mc_est;
    std::cout << "Character ATE:     " << computeATE(est_char, gt.T_mw_gt) << " m\n";

    const double ext_pos_err =
        (T_mc_est.translation() - gt.T_mc_gt.translation()).norm();
    const double ext_rot_err =
        computeRotError(T_mc_est.rotationMatrix(), gt.T_mc_gt.rotationMatrix());
    std::cout << "Extrinsic t error: " << ext_pos_err << " m\n";
    std::cout << "Extrinsic R error: " << ext_rot_err << " deg\n";
    std::cout << "Est T_mc t: [" << T_mc_est.translation().transpose() << "]^T\n";
    std::cout << "True T_mc t: [" << gt.T_mc_gt.translation().transpose() << "]^T\n";

    std::cout << "\nPer-frame camera position errors:\n";
    for (int i = 0; i < N; i += 5) {
        const double e = (est_poses[i].translation() -
                          gt.T_cw_gt[i].translation()).norm();
        std::cout << "  frame " << i << ": " << e << " m\n";
    }

    std::cout << "\n====================================================\n";
    std::cout << "  Done.\n";
    std::cout << "====================================================\n";
    return 0;
}


