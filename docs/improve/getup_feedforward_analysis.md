# GetUp 力矩前馈补偿：架构、验证与结论

## 1. 结论（先行的）

### 1.1 正确的架构

```
q̈_des = Kp·e - Kd·q̇                        (1) PD → 期望加速度, rad/s²
τ_ff  = M_diag·q̈_des + g(q)                 (2) 对角惯量 → 前馈力矩, Nm
gap   = q̈_des - q̈_meas                       (3) 加速度残差, rad/s²
tc   += η·M_diag·gap  (限幅, 延迟启动)       (4) 力矩域迭代补偿, Nm
τ     = τ_ff + tc                            (5) 总力矩, Nm
```

- **PD 重新设计**: Kp = ω_n² (s⁻²), Kd = 2ζω_n (s⁻¹), 输出直接是角加速度 (rad/s²)。
- **M_diag**: 目标姿态质量矩阵对角元, 离线一次计算, 常量。
- **g(q)**: Pinocchio 精确重力, 已验证 g_pin = qbias_MJ (max|Δ|=0)。
- **迭代补偿**: tc 累积加速度残差, 稳态下自动抵消接触力 qc。需延迟启动 (沉降完成后再更新) 和限幅 (防瞬态污染)。

### 1.2 关键验证

supine 姿态, shoulder_pitch 目标 = 0.58 rad (比原目标后退 5°, 肘部适度离地):

| | baseline (PD+g) | **iter (PD+g+tc)** |
|------|:---:|:---:|
| L shoulder err | +2.4° | **-0.2°** |
| R shoulder err | +2.2° | **+0.2°** |
| tc (收敛值) | — | L=+23, R=+22 Nm |
| qc (接触力) | -0.5 Nm | -22.5 Nm |
| 力矩饱和 | 否 | 否 |

- tc 左右对称 (±1 Nm), 误差左右对称 (±0.2°)。
- max err (全受控关节) = 3.0°, 由其他关节主导 (非 shoulder)。
- **迭代补偿使 shoulder 误差从 +2.3° → ±0.2°, 接近零。**

> 注: shoulder_pitch=0.67 rad (原始目标) 时肘部过度陷入地面, 接触力 (~29 Nm) 超过执行器极限 (25 Nm), 力矩饱和 → tc 发散。这不是算法问题, 是执行器能力外的物理约束。后退 5° 消除饱和后系统即正常收敛。

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
| RNEA + a_base=IMU | IMU 含控制反馈, 闭环发散 |
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
