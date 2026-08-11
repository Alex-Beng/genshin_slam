#ifndef SLAM_EKF_SLAM_H
#define SLAM_EKF_SLAM_H

#include "slam/types.h"
#include <vector>
#include <deque>

namespace slam {

struct EKFState {
    SE3 T_cw;        // camera pose in world frame
    SE3 T_mc;        // character-to-camera extrinsic (M -> C)
    Matrix12 P;      // error-state covariance

    EKFState() : T_cw(), T_mc(), P(Matrix12::Identity()) {}
};

// Error-state EKF fusing visual odometry (VO) with minimap observations.
// State: T_cw (camera pose), T_mc (extrinsic), 12-D error state (right perturbation).
class EKFSLAM {
public:
    EKFSLAM();

    void init(const SE3& T_cw_init, const SE3& T_mc_init,
              const Matrix12& P_init = Matrix12::Zero());

    // VO-driven prediction: T_cw <- T_cw * delta_T, T_mc unchanged.
    void predict(const SE3& delta_T, const Matrix12& Q);

    // Minimap absolute observation update: obs = [x_w, z_w, theta_w].
    void update(const MapObs& obs, const Matrix3& R);

    // Minimap frame-to-frame odometry update.
    // delta_world: measured world-frame character displacement (dx, dz, dyaw)
    // between the previous and current camera frames.
    void updateMinimapOdom(const Eigen::Vector3d& delta_world, const Matrix3& R);

    // Reset the "previous frame" reference (e.g. after an occlusion gap).
    void resetOdomRef() { prev_pose_set_ = false; }

    // --- Getters ---
    SE3 getCameraPose() const { return state_.T_cw; }
    SE3 getExtrinsic() const  { return state_.T_mc; }
    Matrix12 getCovariance() const { return state_.P; }
    SE3 getCharacterPose() const { return state_.T_cw * state_.T_mc; }

private:
    EKFState state_;
    double prev_x_ = 0.0, prev_z_ = 0.0, prev_yaw_ = 0.0;  // previous char pose
    bool prev_pose_set_ = false;

    // Extract [x_w, z_w, theta_w] from T_mw (Y-UP, forward = +Z local axis).
    MapObs observe(const SE3& T_mw) const;

    // Predicted world-frame character displacement (dx, dz, dyaw) between the
    // stored previous pose and the current character pose.
    Eigen::Vector3d odomPrediction(const SE3& T_cw, const SE3& T_mc) const;

    // Numerical Jacobian of observe w.r.t. 12-D error state (right perturbation).
    Eigen::Matrix<double, 3, 12> computeObsJacobian(const SE3& T_cw,
                                                     const SE3& T_mc) const;

    // Numerical Jacobian of the world-frame odometry residual w.r.t. 12-D error
    // state (right perturbation of current T_cw and T_mc; prev pose fixed).
    Eigen::Matrix<double, 3, 12> computeOdomJacobian(const SE3& T_cw,
                                                      const SE3& T_mc) const;

    static double wrapAngle(double a);
};

} // namespace slam

#endif // SLAM_EKF_SLAM_H