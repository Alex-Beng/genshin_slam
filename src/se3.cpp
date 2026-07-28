#include "slam/se3.h"
#include "slam/types.h"
#include <cmath>

namespace slam {

cv::Mat so3_exp(const cv::Mat& phi) {
    CV_Assert(phi.size() == cv::Size(1, 3) && phi.type() == CV_64F);
    double theta = cv::norm(phi);
    if (theta < 1e-12) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    cv::Mat u = phi / theta;
    cv::Mat u_hat = skew(u);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F)
                + std::sin(theta) * u_hat
                + (1.0 - std::cos(theta)) * u_hat * u_hat;
    return R;
}

cv::Mat so3_log(const cv::Mat& R) {
    CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_64F);
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    double theta = std::acos(std::min(1.0, std::max(-1.0, (trace - 1.0) / 2.0)));

    if (theta < 1e-12) {
        return cv::Mat::zeros(3, 1, CV_64F);
    }
    if (theta < M_PI - 1e-6) {
        cv::Mat diff = R - R.t();
        cv::Mat w = vee(diff);
        return w * (theta / (2.0 * std::sin(theta)));
    }
    // theta near pi: handle degenerate case
    // Use the diagonal elements to find the rotation axis
    cv::Mat w(3, 1, CV_64F);
    double s = std::sin(theta);
    // Find the largest diagonal element
    if (R.at<double>(0,0) > R.at<double>(1,1) && R.at<double>(0,0) > R.at<double>(2,2)) {
        double v = std::sqrt((R.at<double>(0,0) - std::cos(theta)) / (1.0 - std::cos(theta)) + 1e-10);
        w.at<double>(0) = v;
        w.at<double>(1) = (R.at<double>(0,1) + R.at<double>(1,0)) / (2.0 * v * s) * theta;
        w.at<double>(2) = (R.at<double>(0,2) + R.at<double>(2,0)) / (2.0 * v * s) * theta;
    } else if (R.at<double>(1,1) > R.at<double>(2,2)) {
        double v = std::sqrt((R.at<double>(1,1) - std::cos(theta)) / (1.0 - std::cos(theta)) + 1e-10);
        w.at<double>(0) = (R.at<double>(1,0) + R.at<double>(0,1)) / (2.0 * v * s) * theta;
        w.at<double>(1) = v;
        w.at<double>(2) = (R.at<double>(1,2) + R.at<double>(2,1)) / (2.0 * v * s) * theta;
    } else {
        double v = std::sqrt((R.at<double>(2,2) - std::cos(theta)) / (1.0 - std::cos(theta)) + 1e-10);
        w.at<double>(0) = (R.at<double>(2,0) + R.at<double>(0,2)) / (2.0 * v * s) * theta;
        w.at<double>(1) = (R.at<double>(2,1) + R.at<double>(1,2)) / (2.0 * v * s) * theta;
        w.at<double>(2) = v;
    }
    return w;
}

cv::Mat so3_right_jacobian(const cv::Mat& phi) {
    CV_Assert(phi.size() == cv::Size(1, 3) && phi.type() == CV_64F);
    double theta = cv::norm(phi);
    if (theta < 1e-12) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    cv::Mat u = phi / theta;
    cv::Mat u_hat = skew(u);
    cv::Mat J = cv::Mat::eye(3, 3, CV_64F)
                - (1.0 - std::cos(theta)) / theta * u_hat
                + (theta - std::sin(theta)) / theta * u_hat * u_hat;
    return J;
}

cv::Mat so3_right_jacobian_inv(const cv::Mat& phi) {
    CV_Assert(phi.size() == cv::Size(1, 3) && phi.type() == CV_64F);
    double theta = cv::norm(phi);
    if (theta < 1e-12) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    cv::Mat u = phi / theta;
    cv::Mat u_hat = skew(u);
    double half = 0.5;
    double c = (1.0 - std::cos(theta)) / (theta * theta);
    double d = (theta - std::sin(theta)) / (theta * theta * theta);
    // J_inv = I - 0.5*u_hat + (1/(theta^2) - (1+cos(theta))/(2*theta*sin(theta))) * u_hat^2
    // Simplified: I - 0.5*u_hat + (1/theta^2 - c/(theta^2*sinc_helper)) * u_hat^2
    double term = 1.0 / (theta * theta) - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta) + 1e-12);
    cv::Mat J_inv = cv::Mat::eye(3, 3, CV_64F)
                    - half * u_hat
                    + term * u_hat * u_hat;
    return J_inv;
}

cv::Mat se3_exp(const cv::Mat& xi) {
    CV_Assert(xi.size() == cv::Size(1, 6) && xi.type() == CV_64F);
    cv::Mat rho = xi(cv::Rect(0, 0, 1, 3));
    cv::Mat phi = xi(cv::Rect(0, 3, 1, 3));

    double theta = cv::norm(phi);
    cv::Mat R = so3_exp(phi);

    cv::Mat t;
    if (theta < 1e-12) {
        t = rho.clone();
    } else {
        cv::Mat u_hat = skew(phi / theta);
        cv::Mat V = cv::Mat::eye(3, 3, CV_64F)
                    + (1.0 - std::cos(theta)) / (theta * theta) * u_hat
                    + (theta - std::sin(theta)) / (theta * theta * theta) * u_hat * u_hat;
        t = V * rho;
    }

    return makeT(R, t);
}

cv::Mat se3_log(const cv::Mat& T) {
    CV_Assert(T.size() == cv::Size(4, 4) && T.type() == CV_64F);
    cv::Mat R = rot(T);
    cv::Mat t = trans(T);

    cv::Mat phi = so3_log(R);
    double theta = cv::norm(phi);

    cv::Mat rho;
    if (theta < 1e-12) {
        rho = t.clone();
    } else {
        cv::Mat u_hat = skew(phi / theta);
        cv::Mat V_inv = cv::Mat::eye(3, 3, CV_64F)
                        - 0.5 * u_hat
                        + (1.0 / (theta * theta)
                           - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta) + 1e-12))
                          * u_hat * u_hat;
        rho = V_inv * t;
    }

    cv::Mat xi(6, 1, CV_64F);
    rho.copyTo(xi(cv::Rect(0, 0, 1, 3)));
    phi.copyTo(xi(cv::Rect(0, 3, 1, 3)));
    return xi;
}

cv::Mat se3_adj(const cv::Mat& T) {
    CV_Assert(T.size() == cv::Size(4, 4) && T.type() == CV_64F);
    cv::Mat R = rot(T);
    cv::Mat t = trans(T);
    cv::Mat t_hat = skew(t);

    cv::Mat Ad = cv::Mat::zeros(6, 6, CV_64F);
    // Top-left: R
    R.copyTo(Ad(cv::Rect(0, 0, 3, 3)));
    // Top-right: t_hat * R
    cv::Mat tR = t_hat * R;
    tR.copyTo(Ad(cv::Rect(3, 0, 3, 3)));
    // Bottom-right: R
    R.copyTo(Ad(cv::Rect(3, 3, 3, 3)));
    // Bottom-left: 0

    return Ad;
}

cv::Mat se3_inv(const cv::Mat& T) {
    CV_Assert(T.size() == cv::Size(4, 4) && T.type() == CV_64F);
    cv::Mat R = rot(T);
    cv::Mat t = trans(T);
    cv::Mat Rt = R.t();
    cv::Mat tinv = -Rt * t;
    return makeT(Rt, tinv);
}

cv::Mat se3_compose(const cv::Mat& T1, const cv::Mat& T2) {
    return T1 * T2;
}

} // namespace slam