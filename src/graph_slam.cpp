#include "slam/graph_slam.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace slam {

// Whiten by square root of information matrix (via Cholesky).
static Matrix6 sqrt_info6(const Matrix6& info) {
    Eigen::LLT<Matrix6> llt(info);
    return llt.matrixL().transpose();  // J^T J = info
}

static Matrix3 sqrt_info3(const Matrix3& info) {
    Eigen::LLT<Matrix3> llt(info);
    return llt.matrixL().transpose();
}

GraphSLAM::GraphSLAM() {}

int GraphSLAM::addNode(const SE3& T_init) {
    nodes_.push_back(T_init);
    return (int)nodes_.size() - 1;
}

void GraphSLAM::addVOFactor(int i, int j, const SE3& delta_T_ij, const Matrix6& info) {
    Factor f;
    f.type = FactorType::VO;
    f.node_ids = {i, j};
    f.delta_T = delta_T_ij;
    f.info6 = info;
    factors_.push_back(f);
}

void GraphSLAM::addMapFactor(int pose_idx, int ext_idx, const MapObs& obs,
                             const Matrix3& info) {
    Factor f;
    f.type = FactorType::MAP;
    f.node_ids = {pose_idx, ext_idx};
    f.obs = obs;
    f.info3 = info;
    factors_.push_back(f);
}

void GraphSLAM::addMinimapOdomFactor(int i, int ext_idx, int j,
                                     const MapObs& delta, const Matrix3& info) {
    Factor f;
    f.type = FactorType::MAP_ODOM;
    f.node_ids = {i, ext_idx, j};
    f.obs = delta;
    f.info3 = info;
    factors_.push_back(f);
}

void GraphSLAM::addPriorFactor(int idx, const SE3& T_prior, const Matrix6& info) {
    Factor f;
    f.type = FactorType::PRIOR;
    f.node_ids = {idx};
    f.delta_T = T_prior;
    f.info6 = info;
    factors_.push_back(f);
}

void GraphSLAM::evaluate(const Factor& f, Eigen::VectorXd& r) const {
    if (f.type == FactorType::VO) {
        const SE3 Ti = nodes_[f.node_ids[0]];
        const SE3 Tj = nodes_[f.node_ids[1]];
        const SE3 err = Ti.inverse() * Tj * f.delta_T.inverse();
        r.resize(6);
        r = err.log();
    } else if (f.type == FactorType::MAP) {
        const SE3 Tmw = nodes_[f.node_ids[0]] * nodes_[f.node_ids[1]];
        const Eigen::Vector3d t = Tmw.translation();
        const Eigen::Matrix3d R = Tmw.rotationMatrix();
        const double fx = R(0, 2);
        const double fz = R(2, 2);
        const double theta = std::atan2(fx, fz);
        r.resize(3);
        r << t(0) - f.obs(0),
             t(2) - f.obs(1),
             theta - f.obs(2);
        // wrap
        while (r(2) > M_PI) r(2) -= 2.0 * M_PI;
        while (r(2) < -M_PI) r(2) += 2.0 * M_PI;
        r(0) = t(0) - f.obs(0);  // keep x,z as is
        r(1) = t(2) - f.obs(1);
    } else if (f.type == FactorType::MAP_ODOM) {
        const SE3 Ti = nodes_[f.node_ids[0]];
        const SE3 Tmc = nodes_[f.node_ids[1]];
        const SE3 Tj = nodes_[f.node_ids[2]];
        const SE3 mw_i = Ti * Tmc;
        const SE3 mw_j = Tj * Tmc;
        const Eigen::Vector3d ti = mw_i.translation();
        const Eigen::Vector3d tj = mw_j.translation();
        const Eigen::Matrix3d Ri = mw_i.rotationMatrix();
        const Eigen::Matrix3d Rj = mw_j.rotationMatrix();
        const double yi = std::atan2(Ri(0, 2), Ri(2, 2));
        const double yj = std::atan2(Rj(0, 2), Rj(2, 2));
        r.resize(3);
        r << tj(0) - ti(0), tj(2) - ti(2), std::atan2(std::sin(yj - yi), std::cos(yj - yi));
    } else {  // PRIOR
        const SE3 err = f.delta_T.inverse() * nodes_[f.node_ids[0]];
        r.resize(6);
        r = err.log();
    }
}

double GraphSLAM::getChi2() const {
    double chi2 = 0.0;
    for (const auto& f : factors_) {
        Eigen::VectorXd r;
        evaluate(f, r);
        if (f.type == FactorType::VO || f.type == FactorType::PRIOR) {
            chi2 += r.dot(f.info6 * r);
        } else {
            chi2 += r.dot(f.info3 * r);
        }
    }
    return chi2;
}

void GraphSLAM::optimize(int max_iters, double /*lambda_init*/) {
    if (nodes_.empty()) return;

    // Ceres parameter blocks for each node: [qw,qx,qy,qz, x,y,z]
    std::vector<std::array<double, 7>> params(nodes_.size());
    std::vector<double*> param_ptrs(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        params[i] = se3ToParams(nodes_[i]);
        param_ptrs[i] = params[i].data();
    }

    ceres::Problem problem;

    // Manifold: quaternion (normalized) x Euclidean translation
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto* manifold =
            new ceres::ProductManifold<ceres::QuaternionManifold,
                                       ceres::EuclideanManifold<3>>(
                ceres::QuaternionManifold(), ceres::EuclideanManifold<3>());
        problem.AddParameterBlock(param_ptrs[i], 7, manifold);
    }

    for (const auto& f : factors_) {
        if (f.type == FactorType::VO) {
            VOFactorFunctor ff{f.delta_T, sqrt_info6(f.info6)};
            auto* cost = new ceres::AutoDiffCostFunction<VOFactorFunctor, 6, 7, 7>(
                new VOFactorFunctor(ff));
            problem.AddResidualBlock(cost, nullptr,
                                     param_ptrs[f.node_ids[0]],
                                     param_ptrs[f.node_ids[1]]);
        } else if (f.type == FactorType::MAP) {
            MapFactorFunctor ff{f.obs, sqrt_info3(f.info3)};
            auto* cost = new ceres::AutoDiffCostFunction<MapFactorFunctor, 3, 7, 7>(
                new MapFactorFunctor(ff));
            problem.AddResidualBlock(cost, nullptr,
                                     param_ptrs[f.node_ids[0]],
                                     param_ptrs[f.node_ids[1]]);
        } else if (f.type == FactorType::MAP_ODOM) {
            MinimapOdomFactorFunctor ff{f.obs, sqrt_info3(f.info3)};
            auto* cost =
                new ceres::AutoDiffCostFunction<MinimapOdomFactorFunctor, 3, 7, 7, 7>(
                    new MinimapOdomFactorFunctor(ff));
            problem.AddResidualBlock(cost, nullptr,
                                     param_ptrs[f.node_ids[0]],
                                     param_ptrs[f.node_ids[1]],
                                     param_ptrs[f.node_ids[2]]);
        } else {  // PRIOR
            PriorFactorFunctor ff{f.delta_T, sqrt_info6(f.info6)};
            auto* cost = new ceres::AutoDiffCostFunction<PriorFactorFunctor, 6, 7>(
                new PriorFactorFunctor(ff));
            problem.AddResidualBlock(cost, nullptr,
                                     param_ptrs[f.node_ids[0]]);
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = max_iters;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-12;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Write back
    for (size_t i = 0; i < nodes_.size(); ++i) {
        nodes_[i] = paramsToSE3<double>(params[i].data());
    }
}

void GraphSLAM::marginalize(int idx) {
    // Currently: drop the node and all factors referencing it.
    // NOTE: does not inject the Schur-complement prior (see README findings).
    if (idx < 0 || idx >= (int)nodes_.size()) return;

    factors_.erase(
        std::remove_if(factors_.begin(), factors_.end(),
                       [idx](const Factor& f) {
                           return std::find(f.node_ids.begin(), f.node_ids.end(),
                                            idx) != f.node_ids.end();
                       }),
        factors_.end());

    nodes_.erase(nodes_.begin() + idx);

    // Re-index surviving factor node ids (not performed by old code)
    for (auto& f : factors_) {
        for (auto& nid : f.node_ids) {
            if (nid > idx) --nid;
        }
    }
}

} // namespace slam