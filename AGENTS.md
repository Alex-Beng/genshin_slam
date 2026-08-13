# AGENTS.md — Genshin SLAM

## 项目状态
中期。构建系统、EKF 与图优化 mock 测试已就绪；无 CI、无独立测试框架。

## 项目目标
融合 SLAM 定位系统，结合 Genshin Impact（3D 开放世界游戏）中的小地图数据与单目摄像头输入。融合方式（方案 A）：小地图提供的 2D 位姿 `(x, z, yaw)` 作为全局观测，与视觉里程计（VO）相对位姿融合，在线联合估计相机位姿 `T_cw` 与相机-角色外参 `T_mc`。

## 数学库（取代原手写 se3.cpp）
- **Sophus** — SE(3) 类型（`SE3d::exp` / `log` / `Adj`），右乘扰动约定，Y-UP
- **Eigen** — 矩阵 / 向量运算
- **Ceres Solver 2.2** — 图优化 LM 求解器，`AutoDiffCostFunction` 自动求导（取代手写数值雅可比）
- 头文件：`#include <sophus/se3.hpp>`，`#include <Eigen/Core>`，`#include <ceres/ceres.h>`

## 构建
- 目标平台：Windows，MinGW-w64（GCC 13.2），**C++17**（Ceres 2.2 要求）
- 依赖均位于 `D:\Path`：
  - `Eigen3/include/eigen3`、`sophus/include`（header-only）
  - `Ceres/include` + `Ceres/lib/libceres.a`
  - `glog/include` + `glog/lib/libglog.dll.a`
  - `SuiteSparse/include/CXSparse` + `lib/CXSparse/libcxsparse.dll.a`
- 消费者必须定义 `SOPHUS_USE_BASIC_LOGGING=1` 和 `GLOG_USE_GLOG_EXPORT=1`
- OpenCV 已从核心库移除（仅标定目录可选用）
- 运行前将 `D:\Path\glog\bin` 加入 PATH 或复制 DLL

## 源代码结构
- `src/calibration/` — 摄像头标定（C++ / Python），`test.py` 含硬编码 Linux 路径
- `src/vo_frontend.cpp` + `include/slam/vo_frontend.h` — VO 前端（ORB 匹配 + 本质矩阵位姿估计）
- `src/ekf_slam.cpp` + `include/slam/ekf_slam.h` — 误差状态 EKF（Sophus/Eigen）
- `src/graph_slam.cpp` + `include/slam/graph_slam.h` — Ceres 因子图
- `include/slam/types.h` — 常用类型别名（SE3、Matrix6/12、MapObs、SE2）
- **小地图定位（cvAutoTrack 移植）**
  - `include/slam/capture.h` + `src/capture.cpp` — Capture 抽象层（VideoCapture + BitBlt 窗口截屏，DPI 感知，运行时切换）
  - `include/slam/minimap_pipeline.h` + `src/minimap_pipeline.cpp` — 小地图 ROI 裁剪（1920x1080 基准 MiniMapRect {59,15,218,218}）
  - `include/slam/map_store.h` + `src/map_store.cpp` — 大地图分块 SURF 特征缓存（output.png 12288²→3072² gray，分块+边界扩展+自研二进制缓存 file）
  - `include/slam/surf_loc.h` + `src/surf_loc.cpp` — SURF 小地图→大地图全局定位（BF + 逐轴线性尺度模型 + RANSAC 式剔除，误差 ~2px）
  - `include/slam/minimap_odom.h` + `src/minimap_odom.cpp` — 小地图帧间里程计（ORB 位移 + 极坐标相位相关测角）
  - `src/slam_main.cpp` — 全管线（ROI → SURF 定位 → ORB 里程计 → EKF）
  - `src/capture_test.cpp` / `map_test.cpp` / `surf_test.cpp` / `cache_check.cpp` — 各模块验证
- `src/mock_test.cpp` / `graph_test.cpp` / `vo_test.cpp` — 后端与 VO 测试
- `docs/derivation.md` — 公式推导（EKF + 图优化）

## 小地图定位要点
- 世界坐标 = 大地图 output.png 像素坐标；当前工作图降采样 4x（scale 0.25），`store.scale()` 换算
- SURF 定位输出工作像素；`fitScaleOffset` 用逐轴线性模型 `p_min = A*p_map - B`（注意 B 符号：`B = A*mean(map) - mean(min)`）
- 特征缓存存于 `bigmap_features.bin`（9305 特征，首次提取 ~5s，后续秒载）
- 帧间里程计位移（minimap px）× SURF 拟合尺度（map px / minimap px）= 地图位移
- 测试视频 `F:\Backup\录屏\原神 2022-01-03 11-14-53.mp4`（全程含小地图）
- 大地图 `C:\Users\Alex Beng\Downloads\output.png`

## 约定
- SE(3)：右乘扰动；坐标系 Y-UP；小地图观测 `[x, z, yaw]`，yaw = atan2(R(0,2), R(2,2))
- 小地图帧间里程计用**世界系差分** `[Δx, Δz, Δyaw]`（与 T_mc 无关），yaw 差分需角度归一化
- 所有因子残差含 `sqrt_info` 白化（`evaluate` 用 info 二次型）
- 顶部新增符号/依赖时同步更新 `AGENTS.md`

## 关键命令
```powershell
# 配置
cd build && cmake .. -G "MinGW Makefiles"

# 构建
mingw32-make -j4

# 运行 mock 测试
$env:Path = "D:\Path\glog\bin;$env:Path"
.\build\mock_test.exe     # EKF
.\build\graph_test.exe    # 图优化

# 运行 VO 视频测试（需 OpenCV DLL）
$env:Path = "D:\Path\glog\bin;D:\Path\opencv\x64\mingw\bin;$env:Path"
.\build\vo_test.exe [视频路径]

# 小地图定位模块测试
.\build\capture_test.exe   # ROI 裁剪验证
.\build\map_test.exe       # 大地图特征缓存
.\build\surf_test.exe      # SURF 全局定位精度
.\build\slam_main.exe      # 全管线（视频驱动）
```

## 已知问题
- 外参 `T_mc` 的 Y 平移在纯平面轨迹 + 小地图观测下不可观测；测试中靠先验约束（EKF 靠 Q 随机游走，图优化靠外参先验因子）
- `graph_slam::marginalize()` 仍未注入 Schur 补先验（滑动窗口未启用）
- `SurfLocator::locateNear()` 仍回退全局匹配（连续局部 ROI 匹配未实现；低纹理区域全局匹配会失败）
- `MinimapOdom` 的 yaw 用相位相关近似，`跟随视角`小地图设置下不可靠（cvAutoTrack 同样仅支持 `锁定方向`）

## 前置任务（来自 `todo.md`）
1. 摄像头标定（特征点检测 + subpix，交互式 UI）
2. 运行 SLAM（mask、视频、实时摄像头）