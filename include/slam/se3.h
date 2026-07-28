#ifndef SLAM_SE3_H
#define SLAM_SE3_H

#include <opencv2/opencv.hpp>

namespace slam {

// SE(3) exponential: 6x1 twist vector -> 4x4 transformation matrix
// xi = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]^T
// rho = translation component, phi = rotation component (axis-angle)
cv::Mat se3_exp(const cv::Mat& xi);

// SE(3) logarithm: 4x4 transformation matrix -> 6x1 twist vector
cv::Mat se3_log(const cv::Mat& T);

// SE(3) adjoint: 4x4 -> 6x6
cv::Mat se3_adj(const cv::Mat& T);

// SE(3) inverse: 4x4 -> 4x4
cv::Mat se3_inv(const cv::Mat& T);

// SE(3) compose: T1 * T2
cv::Mat se3_compose(const cv::Mat& T1, const cv::Mat& T2);

// SO(3) exponential: 3x1 axis-angle -> 3x3 rotation matrix
cv::Mat so3_exp(const cv::Mat& phi);

// SO(3) logarithm: 3x3 rotation matrix -> 3x1 axis-angle
cv::Mat so3_log(const cv::Mat& R);

// SO(3) right Jacobian
cv::Mat so3_right_jacobian(const cv::Mat& phi);

// SO(3) inverse right Jacobian
cv::Mat so3_right_jacobian_inv(const cv::Mat& phi);

} // namespace slam

#endif // SLAM_SE3_H