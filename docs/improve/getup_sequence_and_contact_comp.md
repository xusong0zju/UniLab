# G1 起身: 关键姿态序列 + 两阶段接触判断 + 力矩补偿

> 2026-06-28. 基于用户对起身过程的描述整理. 配套图: `logs/getup_keyframes/getup_seq_compare.png`

## 1. 起身过程(5关键帧, 全程双臂协调支撑)

用户描述的起身顺序(非 HumanUP 的反向挺肚子):

```
K1 仰卧双臂支撑 → K2 腿抬起 → K3 蹬腿起身 → K4 蹬腿后中段姿态 → K5 站直
```

**全程双臂辅助支撑**, 协调性关键:

| 帧 | 姿态 | base_z | 双臂 | 腿 | 接触 |
|----|------|--------|------|-----|------|
| K1 | 仰卧背朝下 | 0.12 | 双肩往后(shoulder_pitch正), 双肘接近地面撑地 | 略屈, 准备 | **背+双肘着地** |
| K2 | 仰卧, 腿抬起 | 0.15 | 双肘继续撑地(支点) | 深屈(收腿, 准备蹬) | **背+双肘着地** |
| K3 | 蹬腿起身 | 0.30 | **向上变化**(从肘撑→抬起协调) | 后蹬(hip正), 膝仍弯 | 过渡: 双肘离地→双脚着地 |
| K4 | 蹬腿后中段姿态 | 0.50 | 自然下垂位, 仍协调 | 微屈 | **双脚着地** |
| K5 | 站直 | 0.75 | 双手放下 | 伸直 | **双脚着地** |

**关键纠正**(用户指出):
- "半跪"用词不准. K4 是"蹬腿起身后的中段姿态", 不是传统半跪.
- 全程**双臂辅助支撑**: K1-K2 双臂撑地, K3 双臂向上变化协调(从撑地→抬起), K4-K5 双臂协调放下.
- 起身动力主要靠**腿蹬地**(K3), 双臂提供支撑/平衡协调.

**图**: `logs/getup_keyframes/getup_seq_compare.png`(5帧侧视合成)
- 单帧: `K1_supine_armsupport.png` / `K2_leglift.png` / `K3_kickup.png` / `K4_halfstand.png` / `K5_stand.png`

**手臂接触判断**(实测, 用collision geom而非body原点): G1手臂较短, 但collision geom比body原点外延(半径~0.035m).
- body原点(xpos): 仰卧蓄势时手腕最低 z≈0.065(之前误判够不到地)
- **collision geom最低边**: z≈0.030(差地面3cm, 接近着地)
- 调shoulder角度可能真正接触(≤0). scene_flat.xml只显式配脚-floor contact对(line24-30), 手臂-floor contact未显式配, 故mj_collision未必触发, 但几何上手腕geom接近着地.

**修正**: K1/K2手臂collision geom接近/可能接触地面, 手臂在物理上参与支撑(即使不主动撑高).
后续若需精确判断手/肘接触, 应在scene XML加手-floor contact对, 或用collision geom最低边判据(非body原点).

### 1.1 肘部支撑地面的正确理解(多次修正后的最终结论)

#### 用户提议与工程考量
K1/K2 用肘部着地, 把躯干支撑起来. 两个工程考量:
1. **避免腰撞击**: K3蹬腿时若直接平躺起身, 腰部torso geom落差0.29m(0.050→0.342), 撞击地面风险大. 肘撑起步落差减半, 降低腰撞击.
2. **保护灵巧手**: 肘部着地受力, 力作用在肘部(结实), 不损伤后续安装的灵巧手(精细器件).

#### 关键修正: 朝向与肩部方向(我之前多次搞错)

**修正1: 仰卧 vs 俯卧**
- quat `[cos,0,-sin,0]`(绕y轴-sin) → g_body=[-1,0,0] = **背朝下(仰卧)** ✓(roll_to_supine一致)
- 我之前误渲染成俯卧是视角问题, quat实际是仰卧背朝下.

**修正2: 肩部"往后"的方向**
- 仰卧背朝下时, **shoulder_pitch正值(+2.0)才是"肩往后"**, 肘往下方接近地面(z≈0.004).
- 我之前用负值(-2.0)是"肩往前", 肘朝上不着地. **方向反了**.

**修正3: 用collision geom而非body原点判断着地**
- body原点(xpos): 肘z≈0.09-0.13(误判够不到地)
- **collision geom最低边**: 比body原点低~0.035m(geom半径), 肘geom最低0.001-0.06(接近/着地)

#### 正确姿态(实测验证)

**仰卧(背朝下) + 双肩往后(shoulder_pitch正值) + 双臂对称 + 双肘接近地面:**

```
quat: [cos(45°), 0, -sin(45°), 0]  (仰卧背朝下, tilt90)
shoulder_pitch: +2.0 (双肩往后, 左右相同)
shoulder_roll/yaw: 0 (不外展, 保持对称)
elbow: 0.5 (双肘微弯)
```

偏高5cm落下(锁关节PD)后:
- 左肘 geom 最低 0.071, 右肘 0.060(差6cm, PD锁不住被压弯, 实际GC训练时该能保持)
- 双手腕 z≈0.06(悬空, 保护灵巧手)
- 躯干 z=0.091(被双肘略撑起)

#### 对称性分析(用户质疑, 实测追踪)

**找到不对称根因**: shoulder_roll 的左右符号约定是**符号相反**(镜像):
- shoulder_roll 轴左右都是 [1,0,0](同向), 但左右肩位置 y 镜像
- 所以 shoulder_roll **左右符号相反才镜像对称**(L=+0.2, R=-0.2)
- 若设成**同号(+0.2/+0.2)** → 左右臂往同方向偏, **不对称**(左肘y=+0.11, 右肘y=-0.04)

| 关节类型 | 对称约定 |
|---------|---------|
| pitch类(hip/shoulder/elbow/ankle pitch) | 同号对称(轴[0,1,0]+镜像位置) |
| **roll类(shoulder_roll)** | **符号相反才对称**(轴[1,0,0]同向+镜像位置) |
| yaw类(hip/shoulder yaw) | 同号对称 |

**渲染肘撑姿态时, shoulder_roll 必须设 0/0 或 +L/-R(符号相反), 不能同号.**

追踪落地过程肘部左右位置(shoulder_roll正确设0/0):
| step | 左肘y | 右肘y | 对称? |
|------|-------|-------|------|
| 0(初始) | +0.076 | -0.076 | ✅ 完全对称(镜像) |
| 50 | +0.061 | -0.061 | ✅ 对称 |
| 150 | +0.066 | -0.045 | 开始歪 |
| 300 | +0.152 | +0.034 | ❌ 明显歪 |

**结论**: shoulder_roll符号设对时初始完全对称; 落下过程中被弄歪是锁关节PD太弱 + 接触不对称(动力学问题, 非参数问题).

#### settle后实测数据(sp0.67, 锁关节PD物理稳定)

shoulder_pitch 指令 0.67, settle 后实际关节角(PD锁不住, GC补偿后该无静差):
- shoulder_pitch L/R: 指令+0.67 → **实际+0.49**(被压回, 偏差-0.18)
- elbow L/R: 指令+0.5 → 实际+0.66
- waist_pitch: 指令+0.2 → 实际+0.29

各部位高度(settle后):
- pelvis z=0.073, torso z=0.071(torso-pelvis差-0.002, **背贴地无腾空**)
- 双肘 geom 最低 0.024-0.026(接近地, 差2-3cm)
- 手腕 z=0.099(悬空, 保护灵巧手)

actuator力矩(settle后, PD输出):
- **waist_pitch: 3.67Nm**(腰部最大, 前弯支撑躯干)
- shoulder_pitch L/R: 6.38/5.49Nm(肩撑手臂)
- hip_pitch L/R: 2.38/1.44Nm
- waist_yaw: 1.07Nm, waist_roll: -0.06Nm

**关键**: 这些力矩是 PD 锁关节的输出(无GC). 加GC前馈补偿后, 静差该减小(实际关节角接近指令), 力矩主要由g(q)提供, PD只补动态.
"腰腾空"数据上不存在(torso≈pelvis高度, 背贴地), 渲染视角可能造成误判.

图: `logs/getup_keyframes/supine_elbow_sp067_settled.png`

**关键**: 这些关键帧是**参考轨迹, 非绝对约束帧**.
- 训练时不应该被硬生生带偏成不对称(如落地被歪).
- 逐步撑起来应该好一些(不一次性硬锁, 渐进).
- 训练时robot有自由度, 参考轨迹引导方向, 但允许robot自行协调对称性.

#### 最终判断

K1/K2 用肘撑起上身(仰卧背朝下 + 双肩往后 + 双肘接近地面)是正确方向:
- 减腰撞击 + 保护灵巧手 + 蹬腿更高效
- 双臂对称(初始对称, 落地歪是PD问题, GC训练时该保持)
- 参考轨迹非硬约束, 逐步撑起

**对接触判断**: 阶段1 是**背+双肘着地**(仰卧背着地 + 双肘往后撑地作支点).

---

## 2. 两阶段接触判断(简化)

用户指出: 接触力判断简化为两阶段, 不要复杂多点检测:

### 阶段1: 背+双肘着地(K1-K2)
- **判定**: robot 低位(base_z 低, 如 <0.30) 且 身体倾斜(tilt 大, 接近平躺)
- **支撑基准**: 此时支撑点是**背+双肘**(仰卧背着地 + 双肘往后撑地作支点), 重力经背/肘传地
- **GC 基准**: **pelvis 基准**(当前 g_pelvis, 假设 base 自由)
  - 腿只补腿自身(背/肘支撑了躯干, 腿不需补躯干)
  - **不用 g_foot**(脚没着地, 不是支撑)

### 阶段2: 双脚着地(K4-K5)
- **判定**: robot 抬起, 双脚接触地面(脚 contact 为 True)
- **支撑基准**: 此时支撑点是**双脚**, 重力经腿传地
- **GC 基准**: **足底基准**(g_foot, 腿补含躯干)
  - 腿补含躯干以上全部(躯干14.82kg重力经腿传地, 需 ~12.6Nm 补偿)
  - **不用 g_pelvis**(屁股离地, 不再支撑)

### 关键: 阶段2 "随后即会起身, 不能认为背也着地"
用户强调: 一旦双脚着地(阶段2), 就**不能同时认为背/肘也着地**——
因为双脚着地后 robot 随即起身, 背/肘已离地. 这是**互斥的两阶段**, 不是多点叠加.

→ **GC 基准二选一**: 要么 pelvis(背着地), 要么 foot(双脚着地), 不混合.

---

## 3. 力矩补偿公式

### 3.1 三基准 GC(已实施, 但根据两阶段简化为二选一)

原设计 policy 输出 3 权重(w_foot/w_pelvis/w_hand)连续混合. **根据用户两阶段简化, 改为二选一**:

```
GC = w_pelvis · g_pelvis(q) + w_foot · g_foot(q)    (w_hand 弃用, G1手臂撑不住地)

阶段1(背着地): w_pelvis=1, w_foot=0  → GC = g_pelvis (腿只补腿自身)
阶段2(双脚着地): w_pelvis=0, w_foot=1    → GC = g_foot (腿补含躯干)
```

policy 输出 1 个连续值 w ∈[0,1] 表示阶段过渡(0=阶段1背着地, 1=阶段2双脚着地), 平滑过渡避免跳变.

### 3.2 g_pelvis 与 g_foot 公式

```
g_pelvis(j) = RNEA(q, 0, 0)  [base=pelvis自由, Pinocchio默认]
  → 关节j补偿其pelvis子树下游连杆重力
  → 腿关节: 只补腿自身(hip≈1.6Nm)
  → 躯干经pelvis直接传地, 不在腿的g_pelvis里

g_foot(j) = g_total(j) - g_pelvis(j)  [足底为支撑基准]
  → 关节j补偿足底子树(整个机器人)重力
  → 腿关节: 补含躯干(hip≈13.4Nm, 含躯干14.82kg)
```

**关键恒等式**(支撑链上的关节, 已数值验证误差0):
```
g_total(j) = g_pelvis(j) + g_foot(j)
  ⇒ g_foot(j) = g_total(j) - g_pelvis(j)
```

### 3.3 g_total(整机重力矩, base无关)几何公式

```
g_total(j) = -M_total · (ω_j · cross(p_total - o_j, g_acc))

  M_total = 整机质量 ≈ 33.3 kg
  ω_j     = 关节j的世界系转轴 = oMi[j].rotation @ axis_local[j]
  o_j     = 关节j的世界系原点 = oMi[j].translation
  p_total = 整机COM世界坐标 = centerOfMass(model, data, q)
  g_acc   = [0, 0, -9.81]
```

**实现**: `PinocchioDynamicsModel.gravity_multi_base(qpos, qvel)` 返回 `(g_pelvis, g_total)`.
- g_pelvis: RNEA (computeGeneralizedGravity)
- g_total: 上述几何公式 (centerOfMass + oMi)

### 3.4 完整力矩

```
τ = kp·(q_d - q) - kd·q̇ + GC
  = PD + [w_pelvis·g_pelvis(q) + w_foot·g_foot(q)]
  = PD + [w_pelvis·g_pelvis + w_foot·(g_total - g_pelvis)]   (腿关节)
       [g_pelvis]                                              (非腿关节, 链外)
```

非腿关节(腰/臂)始终用 g_pelvis(躯干/手臂补偿基准不变).

---

## 4. 接触判断实现(两阶段)

```
c_foot = (左脚接触 + 右脚接触) / 2   # 0/0.5/1
阶段判定:
  if c_foot >= 0.5:  阶段2 (双脚着地) → w_foot=1
  else:              阶段1 (背着地) → w_pelvis=1
  (或用 base_z 辅助: base_z<0.30 且 tilt>60° → 阶段1)

policy 输出 w_target ∈[0,1], gc_weight_align reward 引导:
  c = 1 if 阶段2 else 0
  reward = -λ·|w_target - c|
```

**注意**: 不用 w_hand(G1 手臂撑不住地). 之前三基准改二基准(foot/pelvis).

---

## 5. 与 BC 预热的结合

起身序列(K1→K5)可作为 BC 关键帧:
- 用 K1-K5 关节角做 behavior cloning, 让 actor mean 先学会起身序列
- 配合两阶段 GC(BC 时按阶段切 w_pelvis/w_foot)
- BC 预热后 RL 微调

---

## 关键文件
- 图: `logs/getup_keyframes/getup_seq_compare.png` (5帧起身序列)
- 渲染脚本: `/tmp/render_getup_seq.py`
- GC 实施: `src/unilab/control/pinocchio_model.py` (gravity_multi_base), `gravity_comp_controller.py` (gc_weights)
- 接触判断: 待实施(两阶段, 替换原三基准 gc_weight_align)
