# Genshin SLAM 融合定位推导

---

## Phase 1: 坐标系定义与变换链

### 1.1 坐标系

| 符号 | 名称 | 说明 |
|------|------|------|
| W | 世界坐标系 | 游戏世界大地坐标系，右手系，Y-UP |
| C | 相机坐标系 | 单目相机光心，Z 向前，X 向右，Y 向下（OpenCV 惯例） |
| M | 角色/小地图坐标系 | 角色根骨骼坐标系，原点在角色脚底，朝向与游戏朝向一致 |

### 1.2 变换定义

$$
T_{cw} \in SE(3) \quad \text{相机位姿：C $\to$ W，6DOF，待估计}
$$

$$
T_{mc} \in SE(3) \quad \text{外参：M $\to$ C，6DOF，待估计}
$$

合成变换（角色在世界系下的位姿）：

$$
T_{mw} = T_{cw} \circ T_{mc} = T_{cw} \cdot T_{mc} \in SE(3)
$$

参数化方式：

$$
T = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}, \quad
R \in SO(3),\ t \in \mathbb{R}^3
$$

### 1.3 小地图观测

小地图提供角色在 **W 系** 的 2D 位姿观测：

$$
s = \begin{bmatrix} x_w \\ z_w \\ \theta_w \end{bmatrix} \in \mathbb{R}^3
$$

其中 $(x_w, z_w)$ 是水平位置（忽略 Y 轴高度），$\theta_w$ 是偏航角。

从 $T_{mw}$ 提取观测值：

$$
T_{mw} = \begin{bmatrix} R_{mw} & t_{mw} \\ 0^\top & 1 \end{bmatrix}, \quad
R_{mw} = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix}
$$

$$
x_w = t_{mw}[0], \quad z_w = t_{mw}[2], \quad
\theta_w = \text{atan2}(r_{21}, r_{11})
$$

（偏航角提取方式取决于具体旋转约定，此处假设 Y-UP 下的 ZYX 欧拉角顺序）

### 1.4 针孔投影模型

相机内参矩阵 $K$（已标定，已知）：

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

3D 点 $P_w \in \mathbb{R}^3$ 投影到像素 $p \in \mathbb{R}^2$：

$$
p' = K \cdot (R_{cw} \cdot P_w + t_{cw}), \quad
p = \begin{bmatrix} p'_x / p'_z \\ p'_y / p'_z \end{bmatrix}
$$

---

## Phase 2: EKF 版本

### 2.1 状态向量

采用 **误差状态 EKF（ES-EKF）** 处理 $SE(3)$ 状态。

**名义状态**（确定性部分）：

$$
x_{\text{nom}} = \{ T_{cw},\ T_{mc} \}
$$

**误差状态**（扰动部分，用于协方差传播）：

$$
\delta x = \begin{bmatrix} \delta\xi_{cw} \\ \delta\xi_{mc} \end{bmatrix} \in \mathbb{R}^{12}
$$

其中 $\delta\xi \in \mathfrak{se}(3) \cong \mathbb{R}^6$ 是右乘扰动：

$$
\delta\xi = \begin{bmatrix} \delta\rho \\ \delta\phi \end{bmatrix},\quad
\delta\rho \in \mathbb{R}^3 \ (\text{平移}),\quad
\delta\phi \in \mathbb{R}^3 \ (\text{旋转})
$$

**真实状态与名义状态的关系（右乘扰动约定）：**

$$
T_{cw} = T_{cw,\text{nom}} \cdot \exp(\delta\xi_{cw}^\wedge)
$$
$$
T_{mc} = T_{mc,\text{nom}} \cdot \exp(\delta\xi_{mc}^\wedge)
$$

其中 $\exp(\cdot^\wedge): \mathbb{R}^6 \to SE(3)$ 是李指数映射，$(\cdot)^\wedge: \mathbb{R}^6 \to \mathfrak{se}(3)$ 是反对称矩阵。

### 2.2 运动模型

**相机位姿传播（VO 驱动）：**

视觉里程计在 $[k, k+1]$ 间隔内给出相对运动测量 $\Delta T_{k} \in SE(3)$：

$$
T_{cw,k+1} = T_{cw,k} \cdot \Delta T_k
$$

名义状态直接更新：

$$
T_{cw,\text{nom},k+1} = T_{cw,\text{nom},k} \cdot \Delta T_k
$$

**外参传播（随机游走）：**

外参在短时间内视为固定，但缓慢变化以吸收标定漂移：

$$
T_{mc,k+1} = T_{mc,k} \cdot \exp(w_k^\wedge), \quad w_k \sim \mathcal{N}(0, Q_{mc})
$$

名义状态保持不变：

$$
T_{mc,\text{nom},k+1} = T_{mc,\text{nom},k}
$$

**误差状态协方差传播：**

$$
P_{k+1|k} = F_k \cdot P_{k|k} \cdot F_k^\top + Q_k
$$

其中 $F_k$ 是误差状态转移矩阵，$Q_k$ 是过程噪声。

对于 $T_{cw}$ 的误差状态，VO 测量 $\Delta T_k$ 的噪声会通过：

$$
\delta\xi_{cw,k+1} = \text{Ad}(\Delta T_k^{-1}) \cdot \delta\xi_{cw,k} + \text{噪声项}
$$

对于 $T_{mc}$ 的误差状态：

$$
\delta\xi_{mc,k+1} = \delta\xi_{mc,k} + w_k
$$

所以：

$$
F_k = \begin{bmatrix} \text{Ad}(\Delta T_k^{-1}) & 0_{6\times6} \\ 0_{6\times6} & I_6 \end{bmatrix} \in \mathbb{R}^{12\times12}
$$

其中 $\text{Ad}(T) \in \mathbb{R}^{6\times6}$ 是 $SE(3)$ 的伴随矩阵：

$$
\text{Ad}(T) = \begin{bmatrix} R & t^\wedge R \\ 0 & R \end{bmatrix}
$$

### 2.3 观测模型

小地图观测 $s_k = [x_w, z_w, \theta_w]^\top$ 与状态的关系：

$$
T_{mw,k} = T_{cw,k} \cdot T_{mc,k}
$$

观测函数 $h: SE(3) \times SE(3) \to \mathbb{R}^3$：

$$
h(T_{cw}, T_{mc}) = 
\begin{bmatrix}
t_{mw}[0] \\
t_{mw}[2] \\
\text{atan2}(R_{mw}[1,0], R_{mw}[0,0])
\end{bmatrix}
$$

**残差：**

$$
r_k = s_k - h(T_{cw,k}, T_{mc,k}) \in \mathbb{R}^3
$$

**观测雅可比（关键部分）：**

需要计算 $H = \frac{\partial h}{\partial \delta x}\big|_{0}$，即关于误差状态的导数。

对 $T_{cw}$ 右乘扰动 $\delta\xi_{cw}$：

$$
T_{cw}' = T_{cw} \exp(\delta\xi_{cw}^\wedge)
$$
$$
T_{mw}' = T_{cw}' T_{mc} = T_{cw} \exp(\delta\xi_{cw}^\wedge) T_{mc}
$$

利用恒等式 $T^{-1} \exp(\xi^\wedge) T = \exp((\text{Ad}(T^{-1})\xi)^\wedge)$：

$$
T_{mw}' = T_{cw} T_{mc} \cdot T_{mc}^{-1} \exp(\delta\xi_{cw}^\wedge) T_{mc}
= \tilde{T} \cdot \exp((\text{Ad}(T_{mc}^{-1}) \cdot \delta\xi_{cw})^\wedge)
$$

其中 $\tilde{T} = T_{cw} \cdot T_{mc}$。所以 $\delta\xi_{cw}$ 通过 $\text{Ad}(T_{mc}^{-1})$ 映射到 $T_{mw}$ 的右端扰动。

对 $T_{mc}$ 右乘扰动 $\delta\xi_{mc}$：

$$
T_{mc}' = T_{mc} \exp(\delta\xi_{mc}^\wedge)
$$
$$
T_{mw}' = T_{cw} T_{mc} \exp(\delta\xi_{mc}^\wedge) = \tilde{T} \exp(\delta\xi_{mc}^\wedge)
$$

$\delta\xi_{mc}$ 直接映射到 $T_{mw}$ 的右端扰动。

**统一扰动表达：**

$$
T_{mw}(\delta x) = \tilde{T} \cdot \exp\left(\left(J_{mw} \cdot \delta x\right)^\wedge\right)
$$

其中：

$$
J_{mw} = \begin{bmatrix} \text{Ad}(T_{mc}^{-1}) & I_6 \end{bmatrix} \in \mathbb{R}^{6\times12}
$$

**提取观测的雅可比：**

观测函数 $h$ 提取 $T_{mw}$ 的 $[x, z, \theta]$ 分量。令 $T_{mw} = [R|t]$，则：

$$
h(T_{mw}) = \begin{bmatrix} t_0 \\ t_2 \\ \text{atan2}(R_{10}, R_{00}) \end{bmatrix}
$$

对 $T_{mw}$ 右乘扰动 $\delta\xi_{mw} = [\delta\rho, \delta\phi]^\top$ 的雅可比：

$$
\frac{\partial h}{\partial \delta\xi_{mw}}\bigg|_0 = 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{\partial\theta}{\partial\delta\phi_1} & \frac{\partial\theta}{\partial\delta\phi_2} & \frac{\partial\theta}{\partial\delta\phi_3}
\end{bmatrix}_{3\times6}
$$

其中 $\theta = \text{atan2}(R_{10}, R_{00})$，对旋转扰动的导数需要根据具体 $R$ 的值计算。对于 $SO(3)$ 右乘扰动 $R' = R \exp(\delta\phi^\wedge)$，在 $\delta\phi=0$ 处：

$$
\frac{\partial \theta}{\partial \delta\phi} = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix} \cdot R_{2\times2\text{ 相关行}}
$$

（具体表达式依赖于 $R$ 的当前值，在实际实现中数值计算更简单）

**最终观测雅可比矩阵：**

$$
H_k = \frac{\partial h}{\partial \delta x} = \frac{\partial h}{\partial \delta\xi_{mw}} \cdot J_{mw} \in \mathbb{R}^{3\times12}
$$

### 2.4 EKF 更新步骤

**卡尔曼增益：**

$$
K_k = P_{k|k-1} H_k^\top (H_k P_{k|k-1} H_k^\top + R_k)^{-1}
$$

**误差状态更新：**

$$
\delta x_k = K_k \cdot r_k
$$

**名义状态更新（注入误差）：**

$$
T_{cw,\text{nom},k} \leftarrow T_{cw,\text{nom},k} \cdot \exp(\delta\xi_{cw,k}^\wedge)
$$
$$
T_{mc,\text{nom},k} \leftarrow T_{mc,\text{nom},k} \cdot \exp(\delta\xi_{mc,k}^\wedge)
$$

**协方差更新：**

$$
P_{k|k} = (I_{12} - K_k H_k) P_{k|k-1}
$$

**误差状态重置：** 注入后误差状态归零，协方差不变（标准 ES-EKF 做法）。

**实用实现建议：** 观测雅可比 $H_k$ 可通过数值微分直接计算，避免解析推导 $\frac{\partial \theta}{\partial \delta\phi}$。对每个误差状态维度施加小扰动 $\epsilon$，计算 $h$ 的变化量，代码实现更简单且不易出错。

### 2.5 小地图延迟处理

小地图观测可能有显著延迟（网络/渲染流水线延迟）。方案：

在 EKF 中维护一个**状态队列**，当延迟的观测到达时，将观测与对应时刻的状态对齐做更新，然后重新传播到当前时刻。

延迟量 $\tau$ 已知（或在线估计），观测 $s_k$ 实际对应 $k-\tau$ 时刻：

$$
r_k = s_k - h(T_{cw,k-\tau}, T_{mc,k-\tau})
$$

---

## Phase 3: 图优化版本

### 3.1 因子图结构

**节点：**

| 节点 | 维度 | 数量 | 说明 |
|------|------|------|------|
| $T_{cw,i}$ | $SE(3)$ | $N$ | 每个关键帧的相机位姿 |
| $T_{mc}$ | $SE(3)$ | $1$ | 共享外参（全局唯一） |

**因子：**

| 因子 | 连接 | 残差维度 | 说明 |
|------|------|----------|------|
| VO 帧间约束 | $T_{cw,i} - T_{cw,i+1}$ | 6 | 相对位姿测量 |
| 小地图观测 | $T_{cw,i} - T_{mc}$ | 3 | 绝对位姿约束 |
| 先验因子 | $T_{cw,0}$ | 6 | 固定 Gauge 自由度 |

### 3.2 残差定义

#### VO 帧间约束

给定 VO 测量 $\Delta T_{i,i+1} \in SE(3)$，残差定义为相对位姿误差的切空间坐标：

$$
r_{vo}(T_{cw,i}, T_{cw,i+1}) = \log\left( \Delta T_{i,i+1}^{-1} \cdot T_{cw,i}^{-1} \cdot T_{cw,i+1} \right)^\vee \in \mathbb{R}^6
$$

其中 $\log(\cdot)^\vee: SE(3) \to \mathbb{R}^6$ 是李对数映射。

**雅可比（右乘扰动）：**

令 $Z = \Delta T_{i,i+1}^{-1} \cdot T_{cw,i}^{-1} \cdot T_{cw,i+1}$，则 $r_{vo} = \log(Z)^\vee$。

对 $T_{cw,i}$ 右乘扰动 $\delta\xi_i$：

$$
\frac{\partial r_{vo}}{\partial \delta\xi_i} = -\mathcal{J}_r(Z)^{-1} \cdot \text{Ad}(T_{cw,i+1}^{-1} \cdot T_{cw,i})
$$

对 $T_{cw,i+1}$ 右乘扰动 $\delta\xi_{i+1}$：

$$
\frac{\partial r_{vo}}{\partial \delta\xi_{i+1}} = \mathcal{J}_r(Z)^{-1}
$$

其中 $\mathcal{J}_r(Z) \in \mathbb{R}^{6\times6}$ 是 $SE(3)$ 在 $Z$ 处的右雅可比。当残差收敛到零时 $Z \to I$，$\mathcal{J}_r(I) = I_6$，此时雅可比简化为：

$$
\frac{\partial r_{vo}}{\partial \delta\xi_i} \approx -\text{Ad}(T_{cw,i+1}^{-1} \cdot T_{cw,i}), \quad
\frac{\partial r_{vo}}{\partial \delta\xi_{i+1}} \approx I_6
$$

#### 小地图观测约束

给定小地图观测 $s_i = [x_w, z_w, \theta_w]^\top$，残差：

$$
r_{map}(T_{cw,i}, T_{mc}) = s_i - h(T_{cw,i} \cdot T_{mc}) \in \mathbb{R}^3
$$

**雅可比：**

与 EKF 中的观测雅可比相同：

$$
\frac{\partial r_{map}}{\partial \delta\xi_{cw,i}} = -\frac{\partial h}{\partial \delta\xi_{mw}} \cdot \text{Ad}(T_{mc}^{-1})
$$
$$
\frac{\partial r_{map}}{\partial \delta\xi_{mc}} = -\frac{\partial h}{\partial \delta\xi_{mw}}
$$

#### 先验约束

固定第一帧，消除 Gauge 自由度：

$$
r_{prior}(T_{cw,0}) = \log(T_{cw,0}^{-1} \cdot T_{cw,0}^{\text{init}})^\vee \in \mathbb{R}^6
$$

### 3.3 优化问题

$$
\min_{T_{cw,0:N}, T_{mc}} \sum_{i=0}^{N-1} \| r_{vo,i} \|^2_{\Sigma_{vo}} + \sum_{i=0}^{N} \| r_{map,i} \|^2_{\Sigma_{map}} + \| r_{prior} \|^2_{\Sigma_{prior}}
$$

使用 **Levenberg-Marquardt** 或 **Gauss-Newton** 迭代求解。

### 3.4 滑动窗口与边缘化

为限制计算量，维护固定大小的滑动窗口（如 $N=10$）。

**边缘化策略：**

当窗口满时，边缘化最老的帧 $T_{cw,0}$：

1. 将与 $T_{cw,0}$ 相连的所有因子（VO、小地图、先验）线性化
2. 使用 **Schur complement** 将 $T_{cw,0}$ 的信息矩阵边缘化到窗口内的其他变量
3. 构建先验因子附加到 $T_{cw,1}$ 上，替换被边缘化的约束

---

## 符号汇总

| 符号 | 说明 |
|------|------|
| $SE(3)$ | 特殊欧几里得群，刚体变换 |
| $\mathfrak{se}(3)$ | $SE(3)$ 的李代数 |
| $\xi^\wedge \in \mathfrak{se}(3)$ | $\mathbb{R}^6$ 到李代数的映射 |
| $\exp(\xi^\wedge) \in SE(3)$ | 李指数映射 |
| $\log(T)^\vee \in \mathbb{R}^6$ | 李对数映射 |
| $\text{Ad}(T) \in \mathbb{R}^{6\times6}$ | $SE(3)$ 伴随矩阵 |
| $T_{ab}$ | 从 $b$ 系到 $a$ 系的变换 |
| $\delta\xi$ | 右乘扰动（误差状态） |