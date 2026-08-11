#ifndef SLAM_GRAPH_SLAM_H
#define SLAM_GRAPH_SLAM_H

#include "slam/types.h"
#include <ceres/ceres.h>
#include <vector>
#include <array>

namespace slam {

enum class FactorType {
    VO,      // relative pose constraint (6-dim residual)
    MAP,     // minimap observation constraint (3-dim residual)
    PRIOR,   // prior pose constraint (6-dim residual)
    MAP_ODOM,  // minimap SE(2) relative odometry (3-dim residual)
};

// A single factor connecting 1 or 2 nodes.
struct Factor {
    FactorType type;
    std::vector<int> node_ids;  // indices into GraphSLAM::nodes_
    SE3     delta_T;            // VO measurement (T_j = T_i * delta_T) or prior pose
    MapObs  obs;                // MAP observation / minimap odom delta [x, z, yaw]
    Matrix6 info6;              // information matrix (VO / PRIOR)
    Matrix3 info3;              // information matrix (MAP / MAP_ODOM)
};

// ============================================================
// Convert SE3 <-> Ceres parameter block [qw, qx, qy, qz, x, y, z]
// ============================================================
inline std::array<double, 7> se3ToParams(const SE3& T) {
    const Eigen::Quaterniond q = T.unit_quaternion();
    const Eigen::Vector3d t = T.translation();
    return {{q.w(), q.x(), q.y(), q.z(), t(0), t(1), t(2)}};
}

template <typename T>
inline Sophus::SE3<T> paramsToSE3(const T* p) {
    Eigen::Quaternion<T> q(p[0], p[1], p[2], p[3]);
    Eigen::Matrix<T, 3, 1> t(p[4], p[5], p[6]);
    return Sophus::SE3<T>(Sophus::SO3<T>(q), t);
}

// Project an SE(3) transform onto the XZ ground plane as SE(2), templated.
template <typename T>
inline Sophus::SE2<T> se3ToSe2(const Sophus::SE3<T>& Tse3) {
    const Eigen::Matrix<T, 3, 3> R = Tse3.rotationMatrix();
    const Eigen::Matrix<T, 3, 1> t = Tse3.translation();
    const T yaw = atan2(R(0, 2), R(2, 2));
    return Sophus::SE2<T>(Sophus::SO2<T>::exp(yaw),
                          Eigen::Matrix<T, 2, 1>(t(0), t(2)));
}

// ============================================================
// Cost functors (AutoDiff-ready, templated on scalar type)
// ============================================================

// VO relative pose factor: r = log(T_i^{-1} T_j * delta_T^{-1})
struct VOFactorFunctor {
    SE3 delta_T;
    Matrix6 sqrt_info;

    template <typename T>
    bool operator()(const T* p_i, const T* p_j, T* residuals) const {
        const Sophus::SE3<T> Ti = paramsToSE3<T>(p_i);
        const Sophus::SE3<T> Tj = paramsToSE3<T>(p_j);
        const Sophus::SE3<T> pred = Ti.inverse() * Tj;
        const Sophus::SE3<T> err = pred * delta_T.template cast<T>().inverse();
        Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residuals);
        r = sqrt_info.template cast<T>() * err.log();
        return true;
    }
};

// Minimap observation factor: r = sqrt_info * (obs - h(T_cw * T_mc))
struct MapFactorFunctor {
    MapObs obs;
    Matrix3 sqrt_info;

    template <typename T>
    bool operator()(const T* p_cam, const T* p_mc, T* residuals) const {
        const Sophus::SE3<T> Tcw = paramsToSE3<T>(p_cam);
        const Sophus::SE3<T> Tmc = paramsToSE3<T>(p_mc);
        const Sophus::SE3<T> Tmw = Tcw * Tmc;

        const Eigen::Matrix<T, 3, 1> t = Tmw.translation();
        const Eigen::Matrix<T, 3, 3> R = Tmw.rotationMatrix();

        // forward vector (local +Z) in world: R * e_z
        const T fx = R(0, 2);
        const T fz = R(2, 2);
        const T theta = atan2(fx, fz);

        Eigen::Matrix<T, 3, 1> pred;
        pred << t(0), t(2), theta;

        Eigen::Matrix<T, 3, 1> r = obs.template cast<T>() - pred;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = sqrt_info.template cast<T>() * r;
        return true;
    }
};

// Prior pose factor: r = sqrt_info * log(T_prior^{-1} * T)
struct PriorFactorFunctor {
    SE3 T_prior;
    Matrix6 sqrt_info;

    template <typename T>
    bool operator()(const T* p, T* residuals) const {
        const Sophus::SE3<T> Tcur = paramsToSE3<T>(p);
        const Sophus::SE3<T> err = T_prior.template cast<T>().inverse() * Tcur;
        Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residuals);
        r = sqrt_info.template cast<T>() * err.log();
        return true;
    }
};

// Minimap frame-to-frame odometry factor (world-frame displacement).
// Connects i (T_cw_i), j (T_cw_j) and the shared extrinsic T_mc.
// Residual: r = sqrt_info * (predicted_world_delta - measured_delta)
// where predicted_world_delta = (dx, dz, dyaw) of character pose between i and j.
struct MinimapOdomFactorFunctor {
    MapObs delta;     // measured world-frame character displacement (dx, dz, dyaw)
    Matrix3 sqrt_info;

    template <typename T>
    bool operator()(const T* p_i, const T* p_mc, const T* p_j, T* residuals) const {
        const Sophus::SE3<T> Ti = paramsToSE3<T>(p_i);
        const Sophus::SE3<T> Tmc = paramsToSE3<T>(p_mc);
        const Sophus::SE3<T> Tj = paramsToSE3<T>(p_j);

        const Sophus::SE3<T> mw_i = Ti * Tmc;
        const Sophus::SE3<T> mw_j = Tj * Tmc;

        const Eigen::Matrix<T, 3, 1> ti = mw_i.translation();
        const Eigen::Matrix<T, 3, 1> tj = mw_j.translation();
        const Eigen::Matrix<T, 3, 3> Ri = mw_i.rotationMatrix();
        const Eigen::Matrix<T, 3, 3> Rj = mw_j.rotationMatrix();

        const T yi = atan2(Ri(0, 2), Ri(2, 2));
        const T yj = atan2(Rj(0, 2), Rj(2, 2));

        Eigen::Matrix<T, 3, 1> pred;
        pred << tj(0) - ti(0), tj(2) - ti(2), yj - yi;
        // wrap yaw delta to [-pi, pi] (differentiable for AutoDiff)
        pred(2) = atan2(sin(pred(2)), cos(pred(2)));

        Eigen::Matrix<T, 3, 1> r = pred - delta.template cast<T>();
        // wrap the yaw residual to keep it in [-pi, pi]
        r(2) = atan2(sin(r(2)), cos(r(2)));

        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = sqrt_info.template cast<T>() * r;
        return true;
    }
};

// ============================================================
// Graph SLAM (Ceres-backed)
// ============================================================
class GraphSLAM {
public:
    GraphSLAM();

    // Add an SE(3) node (camera pose or shared extrinsic). Returns node id.
    int addNode(const SE3& T_init);

    // VO constraint: measurement delta_T means T_j = T_i * delta_T.
    void addVOFactor(int i, int j, const SE3& delta_T_ij, const Matrix6& info);

    // Minimap constraint: obs = [x_w, z_w, theta_w] of character pose T_cw * T_mc.
    void addMapFactor(int pose_idx, int ext_idx, const MapObs& obs, const Matrix3& info);

    // Minimap frame-to-frame odometry between pose i and pose j (shared extrinsic).
    // delta: measured world-frame character displacement (dx, dz, dyaw) i -> j.
    void addMinimapOdomFactor(int i, int ext_idx, int j,
                              const MapObs& delta, const Matrix3& info);

    // Prior constraint on a pose node.
    void addPriorFactor(int idx, const SE3& T_prior, const Matrix6& info);

    // Run Levenberg-Marquardt via Ceres.
    void optimize(int max_iters = 20, double /*lambda_init*/ = 1.0);

    // Marginalize a node (removes its factors; no Schur prior yet — see README).
    void marginalize(int idx);

    int numNodes() const { return (int)nodes_.size(); }
    const SE3& getNode(int idx) const { return nodes_[idx]; }
    double getChi2() const;

private:
    std::vector<SE3> nodes_;
    std::vector<Factor> factors_;

    // Compute the 6D/3D residual for a factor in double precision.
    void evaluate(const Factor& f, Eigen::VectorXd& residual) const;
};

} // namespace slam

#endif // SLAM_GRAPH_SLAM_H