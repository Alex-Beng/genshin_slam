#ifndef SLAM_EKF_SLAM_H
#define SLAM_EKF_SLAM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "slam/se3.h"

namespace slam {

struct EKFState {
    cv::Mat T_cw;  // 4x4, camera pose in world frame
    cv::Mat T_mc;  // 4x4, character-to-camera extrinsic
    cv::Mat P;     // 12x12, error-state covariance

    EKFState() : T_cw(cv::Mat::eye(4, 4, CV_64F)),
                 T_mc(cv::Mat::eye(4, 4, CV_64F)),
                 P(cv::Mat::eye(12, 12, CV_64F)) {}
};

struct DelayedObs {
    int frame_idx;       // which frame this observation corresponds to
    cv::Mat obs;         // 3x1: [x_w, z_w, theta_w]
    cv::Mat R;           // 3x3 observation noise covariance
};

class EKFSLAM {
public:
    EKFSLAM();

    // Initialize with camera pose and extrinsic
    void init(const cv::Mat& T_cw_init, const cv::Mat& T_mc_init,
              const cv::Mat& P_init = cv::Mat::eye(12, 12, CV_64F) * 1e-3);

    // Prediction step: VO-driven
    // delta_T: 4x4, relative motion from VO
    // Q: 12x12 process noise covariance
    void predict(const cv::Mat& delta_T, const cv::Mat& Q);

    // Update step: minimap observation
    // obs: 3x1 [x_w, z_w, theta_w]
    // R: 3x3 observation noise covariance
    void update(const cv::Mat& obs, const cv::Mat& R);

    // Update with delayed observation (repropagate after)
    void updateDelayed(const DelayedObs& d_obs,
                       const std::vector<cv::Mat>& delta_T_history);

    // --- Getters ---
    cv::Mat getCameraPose() const { return state_.T_cw; }
    cv::Mat getExtrinsic() const  { return state_.T_mc; }
    cv::Mat getCovariance() const { return state_.P; }
    cv::Mat getCharacterPose() const { return se3_compose(state_.T_cw, state_.T_mc); }

private:
    EKFState state_;

    // Observation function: T_mw -> [x, z, theta]^T
    cv::Mat observe(const cv::Mat& T_mw) const;

    // Numerical Jacobian of observation w.r.t. error state
    cv::Mat computeObsJacobian(const cv::Mat& T_cw, const cv::Mat& T_mc) const;

    // Inject error state into nominal state
    void injectErrorState(const cv::Mat& dxi);
};

} // namespace slam

#endif // SLAM_EKF_SLAM_H