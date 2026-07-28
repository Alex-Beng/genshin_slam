#ifndef SLAM_GRAPH_SLAM_H
#define SLAM_GRAPH_SLAM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace slam {

// === Factor types ===
enum FactorType {
    FACTOR_VO    = 0,  // VO relative pose constraint (6-dim)
    FACTOR_MAP   = 1,  // Minimap observation constraint (3-dim)
    FACTOR_PRIOR = 2,  // Prior constraint (6-dim)
};

// === Factor definition ===
struct Factor {
    FactorType type;
    std::vector<int> node_ids;  // IDs of connected nodes (indices in GraphSLAM::nodes_)
    cv::Mat measurement;        // For VO: 4x4 SE3; for MAP: 3x1 [x,z,θ]; for PRIOR: 4x4 SE3
    cv::Mat sqrt_info;          // Square root information matrix (triu of Cholesky)
    int dim;                    // Residual dimension (6, 3, or 6)
};

// === Graph SLAM ===
class GraphSLAM {
public:
    GraphSLAM();

    // --- Graph construction ---
    // Add a node (SE3 pose). Returns the node ID.
    int addNode(const cv::Mat& T_init);

    // Add a VO constraint between node i and node j
    // delta_T_ij: T_j = T_i * delta_T_ij (measurement)
    // info: 6x6 information matrix
    void addVOFactor(int i, int j, const cv::Mat& delta_T_ij, const cv::Mat& info);

    // Add a minimap observation constraint on node pose_idx (T_cw) and the extrinsic node
    // obs: 3x1 [x_w, z_w, θ_w]
    // info: 3x3 information matrix
    void addMapFactor(int pose_idx, int ext_idx, const cv::Mat& obs, const cv::Mat& info);

    // Add a prior constraint on a node
    // T_prior: prior SE3 value
    // info: 6x6 information matrix
    void addPriorFactor(int idx, const cv::Mat& T_prior, const cv::Mat& info);

    // --- Optimization ---
    // Run Levenberg-Marquardt optimization
    void optimize(int max_iters = 20, double lambda_init = 1.0);

    // --- Marginalization (sliding window) ---
    // Marginalize out a node, converting its information into a prior on remaining nodes
    // Returns the Schur-complement prior factor (or empty if not enough connections)
    void marginalize(int idx);

    // --- Getters ---
    int numNodes() const { return (int)nodes_.size(); }
    cv::Mat getNode(int idx) const { return nodes_[idx].clone(); }
    std::vector<cv::Mat> getAllNodes() const { return nodes_; }
    double getChi2() const;

private:
    std::vector<cv::Mat> nodes_;           // SE3 transforms, last node is always T_mc
    std::vector<Factor> factors_;
    int ext_node_id_;                      // ID of the extrinsic node (set on first addNode)

    // --- Internal helpers ---
    // Compute residual for a factor
    cv::Mat evaluateFactor(const Factor& f, const std::vector<cv::Mat>& nodes) const;

    // Compute numerical Jacobian for a factor w.r.t. a specific node
    // Returns: dim x 6 matrix
    cv::Mat numericalJacobian(const Factor& f, const std::vector<cv::Mat>& nodes,
                              int local_node_idx) const;

    // Build the linear system H * dx = -b
    // H: 6M x 6M, b: 6M x 1, where M = numNodes()
    void buildSystem(cv::Mat& H, cv::Mat& b) const;

    // Apply update vector dx (6M x 1) to nodes using right perturbation
    void applyUpdate(const cv::Mat& dx);
};

} // namespace slam

#endif // SLAM_GRAPH_SLAM_H