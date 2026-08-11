#ifndef SLAM_TYPES_H
#define SLAM_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>

namespace slam {

// SE(3) / SO(3) types (double precision)
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

// SE(2) / SO(2) types (for minimap relative odometry)
using SE2 = Sophus::SE2d;
using SO2 = Sophus::SO2d;

// Eigen vector / matrix aliases
using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;

// Minimap observation: [x_w, z_w, theta_w]^T
using MapObs = Vector3;

// Convenience: SE3 from rotation + translation
inline SE3 makeT(const Matrix3& R, const Vector3& t) {
    return SE3(SO3(R), t);
}

// Extract translation from SE3
inline Vector3 trans(const SE3& T) { return T.translation(); }

// Extract rotation from SE3
inline Matrix3 rot(const SE3& T) { return T.rotationMatrix(); }

// Project an SE(3) transform onto the ground plane (XZ) as SE(2).
// Drops the Y translation and extracts yaw about the Y axis.
inline SE2 se3ToSe2(const SE3& T) {
    const Matrix3 R = T.rotationMatrix();
    const Eigen::Vector3d t = T.translation();
    // Yaw from forward vector (local +Z) projected onto XZ plane.
    const double yaw = std::atan2(R(0, 2), R(2, 2));
    SO2 so2 = SO2::exp(yaw);  // note: SO2::exp takes the scalar angle
    return SE2(so2, Eigen::Vector2d(t(0), t(2)));
}

} // namespace slam

#endif // SLAM_TYPES_H