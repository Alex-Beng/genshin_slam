- [x] 后端融合算法
  - [x] SE(3) 数学库（Sophus 替代手写 se3.cpp）
  - [x] 误差状态 EKF（VO + 小地图绝对观测 + 小地图帧间里程计）
  - [x] 图优化（Ceres AutoDiff，VO + 小地图绝对 + 帧间里程计 + 先验因子）
  - [x] Mock 测试（200 帧圆形轨迹）
  - [x] 公式推导文档（docs/derivation.md）

- [x] VO 前端
  - [x] 特征提取（ORB）
  - [x] 帧间特征匹配（BFMatcher + Lowe ratio）
  - [x] 本质矩阵 RANSAC + 位姿恢复
  - [x] 视频驱动 VO → EKF 管线（vo_test.cpp）

- [x] 小地图接入（cvAutoTrack 移植）
  - [x] Capture 抽象层（VideoCapture + BitBlt 窗口截屏，DPI 感知）
  - [x] 小地图 ROI 裁剪（1920x1080 基准 MiniMapRect，任意分辨率比例缩放）
  - [x] 大地图分块 SURF 特征缓存（map_store：12288² 分块+边界扩展+有效区域过滤+自研二进制缓存）
  - [x] SURF 全局定位（surf_loc：BF 匹配 + 逐轴线性尺度模型 + 一致性剔除，误差 ~2px）
  - [x] 小地图帧间里程计（minimap_odom：ORB 两帧差分位移 + 极坐标相位相关测角）
  - [x] slam_main 全管线（ROI → SURF 定位 → ORB 里程计 → EKF 融合）
  - [x] capture_test / map_test / surf_test / cache_check 验证

- [ ] 摄像头标定
  - [ ] 特征点检测 + subpix
  - [ ] 交互式标定 UI
  - [ ] 内参输出（K, dist_coeffs）

- [ ] 运行管线增强
  - [ ] BitBlt 实时模式联调（需运行中原神窗口）
  - [ ] SURF 连续模式（locateNear：上帧 pos 附近局部 ROI 匹配，提速）
  - [ ] 大地图原始分辨率方向（当前降采样 4x，SURF 特征 ~9300）
  - [ ] 可视化输出（轨迹叠加显示）