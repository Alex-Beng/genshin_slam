#include "slam/ekf_slam.h"
#include "slam/se3.h"
#include "slam/types.h"
#include <cmath>

namespace slam {

EKFSLAM::EKFSLAM() {}

void EKFSLAM::init(const cv::Mat& T_cw_init, const cv::Mat& T_mc_init,
                   const cv::Mat& P_init) {
    state_.T_cw = T_cw_init.clone();
    state_.T_mc = T_mc_init.clone();
    state_.P = P_init.clone();
}

void EKFSLAM::predict(const cv::Mat& delta_T, const cv::Mat& Q) {
    // --- Nominal state propagation ---
    state_.T_cw = se3_compose(state_.T_cw, delta_T);
    // T_mc unchanged (random walk)

    // --- Error-state covariance propagation ---
    // F_k = [Ad(delta_T^{-1}), 0; 0, I_6]
    cv::Mat delta_T_inv = se3_inv(delta_T);
    cv::Mat Ad_dTinv = se3_adj(delta_T_inv);

    cv::Mat F = cv::Mat::eye(12, 12, CV_64F);
    Ad_dTinv.copyTo(F(cv::Rect(0, 0, 6, 6)));

    state_.P = F * state_.P * F.t() + Q;
}

cv::Mat EKFSLAM::observe(const cv::Mat& T_mw) const {
    // Extract [x_w, z_w, theta_w] from T_mw
    cv::Mat t = trans(T_mw);  // 3x1
    cv::Mat R = rot(T_mw);    // 3x3

    // Y-UP: yaw = atan2(forward_x, forward_z)
    // forward vector in world frame: R * [0, 0, 1]^T = [R02, R12, R22]^T
    double f_x = R.at<double>(0, 2);
    double f_z = R.at<double>(2, 2);
    double theta = std::atan2(f_x, f_z);

    cv::Mat obs(3, 1, CV_64F);
    obs.at<double>(0) = t.at<double>(0);
    obs.at<double>(1) = t.at<double>(2);
    obs.at<double>(2) = theta;
    return obs;
}

cv::Mat EKFSLAM::computeObsJacobian(const cv::Mat& T_cw, const cv::Mat& T_mc) const {
    // Numerical Jacobian: perturb each of the 12 error-state dimensions
    cv::Mat H = cv::Mat::zeros(3, 12, CV_64F);
    double eps = 1e-6;

    cv::Mat T_mw = se3_compose(T_cw, T_mc);
    cv::Mat obs0 = observe(T_mw);

    // Perturb T_cw (first 6 dimensions)
    for (int i = 0; i < 6; ++i) {
        cv::Mat dxi = cv::Mat::zeros(6, 1, CV_64F);
        dxi.at<double>(i) = eps;
        cv::Mat T_cw_pert = se3_compose(T_cw, se3_exp(dxi));
        cv::Mat T_mw_pert = se3_compose(T_cw_pert, T_mc);
        cv::Mat obs_pert = observe(T_mw_pert);
        cv::Mat diff = (obs_pert - obs0) / eps;
        diff.copyTo(H(cv::Rect(i, 0, 1, 3)));
    }

    // Perturb T_mc (last 6 dimensions)
    for (int i = 0; i < 6; ++i) {
        cv::Mat dxi = cv::Mat::zeros(6, 1, CV_64F);
        dxi.at<double>(i) = eps;
        cv::Mat T_mc_pert = se3_compose(T_mc, se3_exp(dxi));
        cv::Mat T_mw_pert = se3_compose(T_cw, T_mc_pert);
        cv::Mat obs_pert = observe(T_mw_pert);
        cv::Mat diff = (obs_pert - obs0) / eps;
        diff.copyTo(H(cv::Rect(6 + i, 0, 1, 3)));
    }

    return H;
}

void EKFSLAM::update(const cv::Mat& obs, const cv::Mat& R) {
    // --- Compute residual ---
    cv::Mat T_mw = se3_compose(state_.T_cw, state_.T_mc);
    cv::Mat obs_pred = observe(T_mw);
    cv::Mat r = obs - obs_pred;  // 3x1

    // Normalize theta to [-pi, pi]
    while (r.at<double>(2) > M_PI)  r.at<double>(2) -= 2.0 * M_PI;
    while (r.at<double>(2) < -M_PI) r.at<double>(2) += 2.0 * M_PI;

    // --- Compute Jacobian ---
    cv::Mat H = computeObsJacobian(state_.T_cw, state_.T_mc);

    // --- Kalman gain ---
    cv::Mat S = H * state_.P * H.t() + R;
    cv::Mat K = state_.P * H.t() * S.inv(cv::DECOMP_SVD);

    // --- Error-state update ---
    cv::Mat dxi = K * r;  // 12x1

    // --- Inject error state into nominal state ---
    injectErrorState(dxi);

    // --- Covariance update ---
    cv::Mat I = cv::Mat::eye(12, 12, CV_64F);
    state_.P = (I - K * H) * state_.P;
}

void EKFSLAM::injectErrorState(const cv::Mat& dxi) {
    cv::Mat dxi_cw = dxi(cv::Rect(0, 0, 1, 6));
    cv::Mat dxi_mc = dxi(cv::Rect(0, 6, 1, 6));

    cv::Mat dT_cw = se3_exp(dxi_cw);
    cv::Mat dT_mc = se3_exp(dxi_mc);

    state_.T_cw = se3_compose(state_.T_cw, dT_cw);
    state_.T_mc = se3_compose(state_.T_mc, dT_mc);
}

void EKFSLAM::updateDelayed(const DelayedObs& d_obs,
                             const std::vector<cv::Mat>& delta_T_history) {
    // For now, simple implementation: store a history of states and
    // rewind, update, then repropagate.
    // In a full implementation, this would use a state buffer.
    // Simplified: just use the current state (assumes small delay)
    update(d_obs.obs, d_obs.R);
}

} // namespace slam