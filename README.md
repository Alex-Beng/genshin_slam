# Genshin SLAM

融合 SLAM 定位系统：结合 Genshin Impact 小地图数据与单目摄像头输入。

核心模块（`src/se3.cpp`、`src/ekf_slam.cpp`、`src/graph_slam.cpp`）全部自研实现，未使用 Eigen / Ceres / g2o / GTSAM / Sophus 等优化库；OpenCV 仅作为矩阵容器与底层数值原语（`cv::Mat`、`cv::solve`、`cv::SVD`）。

## 已知问题（Known Issues）

### 正确性 bug（高优先级）

1. **`marginalize()` 丢弃边缘化信息** — `src/graph_slam.cpp:309-326`
   - Schur 补已算出 `H_schur / b_schur`，但代码明确注释 "For now, we just skip this"，随后 `:316` 删除涉及该节点的所有因子，却**不注入等价先验因子**。
   - 后果：滑动窗口实际上在丢弃历史信息，误差只累积不消除，且系统不一致。

2. **边缘化后因子索引错位** — `src/graph_slam.cpp:307,316-319`
   - `nodes_.erase()` 使剩余节点 ID 全部左移，但留存因子的 `node_ids` 仍指向旧的全局索引，因子图被破坏（引用错位的节点）。

3. **`updateDelayed()` 为空壳** — `src/ekf_slam.cpp:123-130`
   - 小地图观测存在延迟，但代码直接拿当前状态调用 `update()`，注释自认 "Simplified: just use the current state (assumes small delay)"。
   - 后果：延迟观测被当作即时观测，引入系统性误差。

4. **数值雅可比在 θ 环绕处出错** — `src/graph_slam.cpp:105`、`src/ekf_slam.cpp:51`
   - 固定 `eps=1e-6` 扰动 `atan2` 求导，预测值跨 ±π 时突然跳变 2π，导数方向错误 → 收敛变慢甚至发散。

### 数值 / 鲁棒性

5. EKF 协方差更新非 Joseph 形式（`src/ekf_slam.cpp:109`）— P 会逐渐失去对称性 / 半正定性。
6. 无异常值处理：无 innovation Mahalanobis gating，无 Huber 鲁棒核。真实小地图误检一次即可带崩 EKF 或 LM。
7. 固定 `eps` 数值差分对尺度敏感（旋转与平移量级混合时尤其危险），且无解析雅可比对照校验误差。

### 性能 / 工程

8. `cv::Mat` 密集线性代数：每次运算都堆分配、无表达式模板；图优化每次迭代重建稠密 `6M×6M` 系统并全量数值微分所有因子。滑动窗口一大大即 O(N³) + SVD，实时性受限。
9. 图优化边际化未做优化期 re-linearize（`src/graph_slam.cpp:313,323` 注释已承认），为简化实现。

### 构建 / 依赖

10. 核心库与 `mock_test` / `graph_test` 实际仅依赖 C++ 标准库，但 `CMakeLists.txt` 无条件 `find_package(OpenCV REQUIRED)` 并链接 `${OpenCV_LIBS}` —— OpenCV 为过度依赖。
    - 仅 `src/calibration/detection.cpp` 真正需要 OpenCV，且它**未被加入构建**。
11. 文档约定 OpenCV 位于 `D:\Path\opencv`（MinGW 构建），该路径在目标机器上不存在；且 `find_package` 失败则整个构建失败。
12. `src/calibration/detection.cpp` 使用 `#include <opencv4/opencv2/opencv.hpp>`，与约定风格 `<opencv2/opencv.hpp>` 不一致（`opencv4` 为 Linux 风格前缀）。
13. 构建工具链缺 `cmake`（本机仅有 MinGW g++ 16.1.0 / mingw32-make 4.4.1）。
14. `src/calibration/test.py` 含硬编码 Linux 绝对路径，修改前需先更新（见 AGENTS.md）。