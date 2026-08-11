# Genshin SLAM

融合 SLAM 定位系统：结合 Genshin Impact 小地图数据与单目摄像头输入。

融合方式（方案 A）：小地图提供角色 2D 位姿 `(x, z, yaw)` 作为全局观测，与视觉里程计（VO）相对位姿融合，在线联合估计相机位姿 `T_cw ∈ SE(3)` 与相机-角色外参 `T_mc ∈ SE(3)`。

数学由 **Sophus**（SE(3) 类型）、**Eigen**（线性代数）、**Ceres 2.2**（图优化 LM + 自动求导）提供，核心 SLAM 代码不再依赖 OpenCV。

## 模块

| 模块 | 文件 | 说明 |
|------|------|------|
| SE(3) 类型 | `sophus` | `SE3d::exp/log/Adj`，右乘扰动，Y-UP |
| 误差状态 EKF | `src/ekf_slam.cpp` | 12 维误差状态，VO 驱动预测，小地图绝对观测 + 帧间里程计更新 |
| 图优化 SLAM | `src/graph_slam.cpp` | Ceres 因子图：VO + 小地图绝对观测 + 小地图帧间里程计 + 先验，`AutoDiffCostFunction` |
| EKF 测试 | `src/mock_test.cpp` | 圆形轨迹 + 噪声，200 帧 |
| 图优化测试 | `src/graph_test.cpp` | 同上，200 帧批量优化 |
| 推导文档 | `docs/derivation.md` | EKF + 图优化完整公式推导 |

## 构建与运行

```powershell
# 配置（依赖在 D:\Path，MinGW-w64 + CMake）
cd build && cmake .. -G "MinGW Makefiles"

# 构建
mingw32-make -j4

# 运行（需 D:\Path\glog\bin 在 PATH）
$env:Path = "D:\Path\glog\bin;$env:Path"
.\build\mock_test.exe     # EKF
.\build\graph_test.exe    # 图优化
```

依赖：Eigen3、Sophus（header-only）、Ceres 2.2（`libceres.a`）、glog、SuiteSparse CXSparse，均位于 `D:\Path`。消费者需定义 `SOPHUS_USE_BASIC_LOGGING=1` 与 `GLOG_USE_GLOG_EXPORT=1`。

## Mock 测试结果

（200 帧圆形轨迹 r=5m，VO 噪声 0.02m / 0.005rad，小地图绝对观测 1Hz（0.15m / 0.05rad），小地图帧间里程计 10Hz（0.02m / 0.01rad））

| 指标 | EKF | 图优化 |
|------|-----|--------|
| Camera ATE | 0.24 m | **0.13 m** |
| Character ATE | 0.25 m | **0.12 m** |
| 外参平移误差 | **0.07 m** | 0.08 m |
| 外参旋转误差 | 1.07 deg | 3.96 deg |

小地图帧间里程计将 EKF 的 Camera ATE 从 0.51m（仅绝对观测）降到 0.24m，图优化从 0.19m 降到 0.13m：它提供与 VO 独立的高频相对约束，显著抑制帧间漂移。

图优化因联合重线性化位姿更准；EKF 顺序处理、内存 O(1)，二者在观测足够时均可收敛。

## 已知问题

- **外参 `T_mc` 的 Y 平移不可观测**：纯平面运动 + `(x,z,yaw)` 小地图观测下，垂直方向无约束。图优化需外参先验因子，EKF 需 Q 随机游走保持可辨识。
- `graph_slam::marginalize()` 尚未注入 Schur 补先验（滑动窗口模式未启用）。
- `docs/derivation.md` 中 EKF 的 `updateDelayed()` 延迟处理仍为简化实现。

## 下一步（见 `todo.md`）

1. 摄像头标定（特征点检测 + subpix，交互式 UI）
2. 运行 SLAM（mask、视频、实时摄像头）