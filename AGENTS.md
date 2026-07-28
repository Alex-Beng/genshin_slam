# AGENTS.md — Genshin SLAM

## 项目状态
早期。尚无构建系统、CI、测试或 lint 配置。

## 项目目标
融合 SLAM 定位系统，结合 Genshin Impact（3D 开放世界游戏）中的小地图数据与单目摄像头输入。

## 构建
- 目标平台：Windows，MinGW-w64（GCC），C++14
- 依赖：OpenCV 4.10.0，安装于 `D:\Path\opencv`（MinGW 构建）
- 尚无构建系统 — 需要先搭建 `CMakeLists.txt` 再构建
  - 使用 `-G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=D:/Path/opencv`
  - C++ 头文件：`#include <opencv2/opencv.hpp>`
- 运行前需将 `D:\Path\opencv\x64\mingw\bin` 加入 PATH 或复制 DLL

## 源代码结构
- `src/calibration/` — 摄像头标定代码（C++ 和 Python 实现）
- `src/calibration/test.py` 包含硬编码的 Linux 绝对路径 — 修改前请先更新
- `src/se3.cpp` — SE(3) 数学库（exp/log/adj/compose）
- `src/ekf_slam.cpp` — 误差状态 EKF 实现
- `src/graph_slam.cpp` — 图优化实现（LM 求解器 + 因子图）
- `src/mock_test.cpp` — EKF mock 测试（圆形轨迹 + 噪声）
- `src/graph_test.cpp` — 图优化 mock 测试（对比 EKF）
- `include/slam/` — 头文件
- `docs/derivation.md` — 融合定位公式推导（EKF + 图优化）

## 约定
- Python 导入风格：`detection.py` 使用 `import cv2`，`test.py` 使用 `import cv2 as cv` —— 保持统一
- C++ 风格：`#include <opencv2/opencv.hpp>`，`using namespace cv;`
- 标定棋盘格尺寸：4x3 内角点
- SE(3) 约定：右乘扰动，Y-UP 坐标系

## 关键命令
```powershell
# 配置
cd build && cmake .. -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=D:/Path/opencv

# 构建
mingw32-make -j4

# 运行 mock 测试
$env:Path = "D:\Path\opencv\x64\mingw\bin;$env:Path"
.\build\mock_test.exe

# 运行图优化 mock 测试
.\build\graph_test.exe
```

## 前置任务（来自 `todo.md`）
1. 摄像头标定（特征点检测 + subpix，交互式 UI）
2. 运行 SLAM（mask、视频、实时摄像头）