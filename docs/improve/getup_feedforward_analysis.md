# GetUp 力矩前馈补偿：架构、验证与结论

## 1. 结论（先行的）

### 1.1 正确的架构

```
q̈_des = Kp·e - Kd·q̇                        (1) PD → 期望加速度, rad/s²
τ_ff  = RNEA(q, q̇, [a_base=0, q̈_des])[6:35] (2) 全动力学 → 前馈力矩, Nm
q̈_meas= (q̇_k - q̇_{k-1}) / h                (3) 编码器差分 → 实测加速度
gap   = q̈_des - q̈_meas                       (4) 加速度残差, rad/s²
tc   += η·M_diag·gap  (限幅, 延迟启动)       (5) 力矩域迭代补偿, Nm
τ     = τ_ff + tc                            (6) 总力矩, Nm
```

- **PD 重新设计**: Kp = ω_n² (s⁻²), Kd = 2ζω_n (s⁻¹), 输出直接是角加速度 (rad/s²)。
- **RNEA 前馈**: $a_{base}=0$（独立预测值, 不依赖 IMU）, 全 M 矩阵包含 base-关节耦合。
- **M_diag**: 目标姿态质量矩阵对角元, 仅用于 tc 的 gap→力矩转换（离线常量）。
- **迭代补偿**: tc 累积加速度残差, 稳态下自动抵消接触力 qc。需延迟启动 + 限幅。

### 1.1.1 RNEA 与 IMU 的角色分离

RNEA 完整形式：$\tau = M(q)·a + C(q,\dot{q})\dot{q} + g(q)$，其中 $a = [a_{base}(6), a_{joint}(29)]$。

**IMU 不能直接填入 RNEA 的 $a_{base}$**——IMU 测量的骨盆加速度是控制力矩 $\tau$ 的**结果**，而 RNEA 的 $a_{base}$ 是**假设的输入条件**。若 $a_{base} := a_{IMU}$，则：

$$\tau = M·[a_{IMU}, q\ddot{}_{des}] + C\dot{q} + g$$

其中 $a_{IMU} = f(\tau)$（$\tau$ 产生的加速度）。$\tau$ 同时出现在等式两边 → 代数闭环 → 发散。验证结果：静态下 RNEA+IMU → **50° 误差 + 力矩饱和**。

正确的分离：

| 角色 | 输入 | 作用 |
|------|------|------|
| **RNEA 前馈** | $a_{base} = 0$ (或离线预测值) + $q\ddot{}_{des}$ | 已知动力学 (重力+惯量+Cori) |
| **IMU 扰动观测** | $q\ddot{}_{IMU}$ (实测关节加速度) | 计算 gap → 驱动 tc |

$$\boxed{\tau = \underbrace{RNEA(q,\dot{q},[0, q\ddot{}_{des}])_{[6:35]}}_{\text{前馈: 已知动力学}} + \underbrace{tc}_{\text{IMU驱动: 未知扰动}}}$$

当前简化版用 $M_{diag}·q\ddot{}_{des} + g(q)$ 代替 RNEA——对角近似, 避免 $a_{base}$ 设置问题。两者的 tc 更新路径相同（加速度残差）。

### 1.2 关键验证

supine, shoulder_pitch=0.58 rad, PD: ω_n=10, ζ=0.7, tc: η=0.10, 延迟2000步, 限幅±5/M_diag。

| 方案 | L shoulder | R shoulder | max err (全关节) | tc_sh L/R | 饱和 |
|------|:---:|:---:|:---:|:---:|:---:|
| PD+g(q) 基线 | +2.4° | +2.2° | 6.3° | — | 否 |
| M_diag + tc | -0.5° | +0.3° | 3.0° | +21/+24 | 否 |
| **RNEA(0) + tc** | **-0.0°** | **-0.0°** | **1.3°** | **+15/+16** | **否** |

- RNEA 使用全 M 矩阵（含 base-关节耦合），比对角近似更精确 → tc 收敛值更小（15 vs 21 Nm）。
- max 误差 1.3° 出现在 wrist_roll 关节（惯量最小, tc 积累最慢），其余全在 ±0.5° 以内。
- 左右完全对称，无饱和。

### 1.3 静差的物理根源

稳态力平衡: $K_p^{eff}·e + qc = 0 \;\Rightarrow\; e = -qc / K_p^{eff}$

其中 $K_p^{eff} = M_{diag}·K_p^{acc}$, $qc$ 是接触力 (不可测)。当接触力超过执行器极限时系统无法进一步逼近目标——这是物理约束, 非控制问题。

---

## 2. 系统模型

### 2.1 Pinocchio 重力补偿

$$g(q) = RNEA(q, 0, 0)_{[6:35]}$$

- quaternion 转换: `[w,x,y,z] → [x,y,z,w]` (直接, 无 conjugate), **正确**。
- 验证: g_pin = qfrc_bias_MJ (max|Δ|=0 Nm)。

### 2.2 全模型动力学

RNEA 完整形式: $\tau = M(q)·a + C(q,\dot{q})\dot{q} + g(q)$

其中 a = [a_base(6), a_joint(29)]。浮基机器人的关键约束: a_base 不可直接设 0 (base 非静止), 也不可用 IMU 实测 (含控制自身反馈)。当前方案用 $M_{diag}$ 对角近似替代完整 M, 避免了这个闭环问题。

---

## 3. 接触力分析 (supine, elbow=0.5, waist 自由)

| 接触点 | 力 ≈ | 影响关节 | qc (基准) | err (Kp_eff=17) |
|--------|:---:|---------|:---:|:---:|
| pelvis | 109N | hip_pitch | ±3.3 Nm | ±4.6° |
| ankle | 44N | knee | ±1.5 Nm | ±0.9° |
| elbow | 54N | shoulder_pitch | ±1.7 Nm | ±6.9° |
| wrist | 12N | elbow | ±0.6 Nm | ±2.3° |

---

## 4. 当前最佳方案

$$\boxed{\tau = M_{diag}·(K_p^{acc}e - K_d^{acc}\dot{q}) + g(q) + tc}$$

| 参数 | 值 | 含义 |
|------|----|------|
| ω_n | 10 rad/s | PD 自然频率 |
| ζ | 0.7 | PD 阻尼比 |
| Kp_acc | 100 s⁻² | = ω_n² |
| Kd_acc | 14 s⁻¹ | = 2ζω_n |
| M_diag | 见 §4.1 | 对角惯量 (离线常量) |
| tc 更新 η | 0.10 | 力矩域迭代增益 |
| tc 更新延迟 | 2000 步 | 避开瞬态 |
| gap 限幅 | ±5/M_diag | 对应接触力 ≤5Nm |

### 4.1 M_diag 取值

目标姿态下 `pin.computeAllTerms` → `np.diag(M)[6:35]`, 各关节近似:

| 关节 | M_diag |
|------|:---:|
| hip_pitch | 0.822 |
| knee | 0.126 |
| shoulder_pitch | 0.172 |
| elbow | 0.044 |
| wrist | 0.010 |

---

## 5. 已排除的错误路径

| 尝试 | 问题 |
|------|------|
| 经验固定修正 (tc 一次测量固定) | qc 跨姿态变化, 不收敛 |
| 几何法 r×F | 浮基吸收接触力, 50× 偏大 |
| forwardDynamics + KKT | λ=0 (刚性 vs 软接触失配) |
| 动量观测器 | 补偿自身污染观测 |
| RNEA + a_base=0 | 浮基不静止, 系统偏差 |
| RNEA + a_base=IMU | IMU 含控制反馈, 闭环发散 | $\tau = M·a_{IMU} + ...$, 但 $a_{IMU}=f(\tau)$, 代数闭环 |
| 无延迟迭代 (tc 从头累加) | 瞬态污染, 发散 |
| 无饱和处理 (tc 无限制) | tc→∞, 力矩饱和后不收敛 |
| 子模型隔离 | Δg≠0 需腿反力, 等价全模型 |

---

## 6. 代码位置

| 文件 | 内容 |
|------|------|
| `src/unilab/control/pinocchio_model.py` | Pinocchio 模型 (use_conjugate_quat=False) |
| `src/unilab/control/modular_torque_controller.py` | use_mj_constraint, use_imu_blend |
| `scripts/verify_getup_torque_comp.py` | 静态验证脚本 |
