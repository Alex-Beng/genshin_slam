#include "slam/ekf_slam.h"
#include <cmath>

namespace slam {

EKFSLAM::EKFSLAM() {}

void EKFSLAM::init(const SE3& T_cw_init, const SE3& T_mc_init,
                   const Matrix12& P_init) {
    state_.T_cw = T_cw_init;
    state_.T_mc = T_mc_init;
    state_.P = P_init;
    prev_pose_set_ = false;
}

double EKFSLAM::wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

MapObs EKFSLAM::observe(const SE3& T_mw) const {
    const Eigen::Vector3d t = T_mw.translation();
    const Eigen::Matrix3d R = T_mw.rotationMatrix();
    // forward vector (local +Z) projected to world: R * e_z
    const double fx = R(0, 2);
    const double fz = R(2, 2);
    MapObs obs;
    obs << t(0), t(2), std::atan2(fx, fz);
    return obs;
}

Eigen::Matrix<double, 3, 12>
EKFSLAM::computeObsJacobian(const SE3& T_cw, const SE3& T_mc) const {
    const double eps = 1e-6;
    Eigen::Matrix<double, 3, 12> H;
    H.setZero();

    const MapObs r0 = observe(T_cw * T_mc);

    for (int i = 0; i < 6; ++i) {
        Vector6 dxi = Vector6::Zero();
        dxi(i) = eps;
        const MapObs rp = observe(T_cw * SE3::exp(dxi) * T_mc);
        H.block<3, 1>(0, i) = (rp - r0) / eps;
    }
    for (int i = 0; i < 6; ++i) {
        Vector6 dxi = Vector6::Zero();
        dxi(i) = eps;
        const MapObs rp = observe(T_cw * T_mc * SE3::exp(dxi));
        H.block<3, 1>(0, 6 + i) = (rp - r0) / eps;
    }
    return H;
}

void EKFSLAM::predict(const SE3& delta_T, const Matrix12& Q) {
    // Save the current character pose as the reference for the next SE(2) odom
    // BEFORE propagation (i.e. this is the pose at the "previous" timestamp).
    if (!prev_pose_set_) {
        prev_x_ = (state_.T_cw * state_.T_mc).translation()(0);
        prev_z_ = (state_.T_cw * state_.T_mc).translation()(2);
        prev_yaw_ = std::atan2((state_.T_cw * state_.T_mc).rotationMatrix()(0, 2),
                               (state_.T_cw * state_.T_mc).rotationMatrix()(2, 2));
        prev_pose_set_ = true;
    }
    // Nominal propagation
    state_.T_cw = state_.T_cw * delta_T;
    // T_mc unchanged (random walk zero-mean)

    // Error-state transition F = [Ad(delta_T^{-1}), 0; 0, I6]
    const Matrix6 ad = delta_T.inverse().Adj();
    Matrix12 F = Matrix12::Identity();
    F.topLeftCorner<6, 6>() = ad;

    state_.P = F * state_.P * F.transpose() + Q;
}

// World-frame character displacement (dx, dz, dyaw) between the previous
// reference pose and the current character pose T_cw * T_mc.
Eigen::Vector3d EKFSLAM::odomPrediction(const SE3& T_cw, const SE3& T_mc) const {
    const SE3 T_mw = T_cw * T_mc;
    const Eigen::Vector3d t = T_mw.translation();
    const Eigen::Matrix3d R = T_mw.rotationMatrix();
    const double yaw = std::atan2(R(0, 2), R(2, 2));
    Eigen::Vector3d d;
    d << t(0) - prev_x_, t(2) - prev_z_, wrapAngle(yaw - prev_yaw_);
    return d;
}

Eigen::Matrix<double, 3, 12>
EKFSLAM::computeOdomJacobian(const SE3& T_cw, const SE3& T_mc) const {
    const double eps = 1e-6;
    Eigen::Matrix<double, 3, 12> H;
    H.setZero();

    const Eigen::Vector3d r0 = odomPrediction(T_cw, T_mc);

    for (int i = 0; i < 6; ++i) {
        Vector6 dxi = Vector6::Zero();
        dxi(i) = eps;
        const Eigen::Vector3d rp = odomPrediction(T_cw * SE3::exp(dxi), T_mc);
        H.block<3, 1>(0, i) = (rp - r0) / eps;
    }
    for (int i = 0; i < 6; ++i) {
        Vector6 dxi = Vector6::Zero();
        dxi(i) = eps;
        const Eigen::Vector3d rp = odomPrediction(T_cw, T_mc * SE3::exp(dxi));
        H.block<3, 1>(0, 6 + i) = (rp - r0) / eps;
    }
    return H;
}

void EKFSLAM::updateMinimapOdom(const Eigen::Vector3d& delta_world,
                                const Matrix3& R) {
    if (!prev_pose_set_) return;

    // Residual: measured minus predicted world-frame displacement.
    Eigen::Vector3d r = delta_world - odomPrediction(state_.T_cw, state_.T_mc);
    r(2) = wrapAngle(r(2));

    const Eigen::Matrix<double, 3, 12> H =
        computeOdomJacobian(state_.T_cw, state_.T_mc);

    const Eigen::Matrix3d S = H * state_.P * H.transpose() + R;
    const Eigen::Matrix<double, 12, 3> K =
        state_.P * H.transpose() * S.inverse();

    const Eigen::Matrix<double, 12, 1> dxi = K * r;

    const Vector6 dxi_cw = dxi.head<6>();
    const Vector6 dxi_mc = dxi.tail<6>();
    state_.T_cw = state_.T_cw * SE3::exp(dxi_cw);
    state_.T_mc = state_.T_mc * SE3::exp(dxi_mc);

    const Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12, 12>::Identity();
    const Eigen::Matrix<double, 12, 12> IKH = I - K * H;
    state_.P = IKH * state_.P * IKH.transpose() + K * R * K.transpose();

    // Retarget the reference to the now-corrected current pose so the next
    // odometry increment is measured against the corrected frame.
    prev_x_ = (state_.T_cw * state_.T_mc).translation()(0);
    prev_z_ = (state_.T_cw * state_.T_mc).translation()(2);
    prev_yaw_ = std::atan2((state_.T_cw * state_.T_mc).rotationMatrix()(0, 2),
                           (state_.T_cw * state_.T_mc).rotationMatrix()(2, 2));
}

void EKFSLAM::update(const MapObs& obs, const Matrix3& R) {
    const SE3 T_mw = state_.T_cw * state_.T_mc;
    const MapObs pred = observe(T_mw);

    MapObs r = obs - pred;
    r(2) = wrapAngle(r(2));

    const Eigen::Matrix<double, 3, 12> H = computeObsJacobian(state_.T_cw, state_.T_mc);

    // Kalman gain
    const Eigen::Matrix3d S = H * state_.P * H.transpose() + R;
    const Eigen::Matrix<double, 12, 3> K =
        state_.P * H.transpose() * S.inverse();

    // Error-state update
    const Eigen::Matrix<double, 12, 1> dxi = K * r;

    // Inject error state (right perturbation)
    const Vector6 dxi_cw = dxi.head<6>();
    const Vector6 dxi_mc = dxi.tail<6>();
    state_.T_cw = state_.T_cw * SE3::exp(dxi_cw);
    state_.T_mc = state_.T_mc * SE3::exp(dxi_mc);

    // Joseph-form covariance update (keeps P symmetric / PSD)
    const Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12, 12>::Identity();
    const Eigen::Matrix<double, 12, 12> IKH = I - K * H;
    state_.P = IKH * state_.P * IKH.transpose() + K * R * K.transpose();

    // The absolute observation also corrects the current pose; retarget the
    // odometry reference to it so the next increment is measured consistently.
    prev_x_ = (state_.T_cw * state_.T_mc).translation()(0);
    prev_z_ = (state_.T_cw * state_.T_mc).translation()(2);
    prev_yaw_ = std::atan2((state_.T_cw * state_.T_mc).rotationMatrix()(0, 2),
                           (state_.T_cw * state_.T_mc).rotationMatrix()(2, 2));
}

} // namespace slam