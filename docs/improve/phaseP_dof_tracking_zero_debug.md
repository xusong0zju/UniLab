# Phase P: dof_tracking 恒 0 的三个根因与修复

> 2026-06-26 Phase P(HumanUP 轨迹 tracking 起身)调试记录。
> 起步时 `reward/dof_tracking` 从训练开始到 iter 数百一直精确为 0,
> 关节追踪完全失效。连续定位并修复了三个独立根因后,dof_tracking 才正常上升。
> 本文记录三个 bug 的诊断过程与修复方法,供后续 motion-tracking 类任务参考。

## 背景

Phase P 设计:fallen env 从仰卧轨迹首帧起步,reward 引导 robot 翻身→仰卧→
跟踪 HumanUP 起身轨迹关节角起身到站立。核心 reward:

- `roll_to_supine`(权重 4):鼓励翻到仰卧(背朝下)
- `dof_tracking`(权重 5):跟踪轨迹当前帧的 29 关节角,`exp(-MSE/σ²)`
- `height_tracking`(权重 3):跟踪轨迹当前帧的 base_z
- `height_exp`/`stand_feet`/`uprightness_exp`:起身终点 reward

`dof_tracking` 是引导 robot 按轨迹关节角起身的关键,但它从一开始就恒 0,
导致 tracking 引导完全失效,robot 靠其他 reward 自由发挥(最终学成翘臀局部最优,
见末节)。三个根因层层叠加,逐一修复后 dof_tracking 才恢复学习信号。

---

## 根因 1:tracking_frame 触发符号反了

### 现象
`dof_tracking` 恒 0,因为 `_tracking_frame` 数组(标记每个 env 当前该 track 哪一帧,
-1=还在翻身未触发,0+=已到仰卧开始 track)从不从 -1 翻成 0。

### 诊断
step() 里判断"是否到仰卧"的代码:

```python
g_body = obs_np[:, 3:6]              # 从 obs 读 gravity
cos_supine = g_body[:, 0] * (-1.0)   # g·[-1,0,0] = -g_x
reached_supine = cos_supine > 0.5
```

逻辑上仰卧(背朝下)时 g_body=[-1,0,0],`g·[-1,0,0]=+1`,cos_supine 应=+1 触发。
dry-run 实测 reset 后 quat 正确(仰卧),但 `cos_supine = -0.95`(不触发)。

**根因**:`obs_np[:, 3:6]` 不是 g_body。看 `joystick.py:_compute_obs`:

```python
actor = np.concatenate([
    noisy_gyro * ...,
    -noisy_gravity,    # ← 取反!obs 里的 gravity = -ctx.gravity
    ...
])
```

obs 里存的 gravity 是 **`-ctx.gravity`**(取反后的)。而 `ctx.gravity`(reward context 用的)
来自 `self._backend.get_sensor_data(self._cfg.sensor.upvector)`(joystick.py:357)。

所以:
- `ctx.gravity`(仰卧)= [-1, 0, 0]
- `obs gravity`(仰卧)= -ctx.gravity = **[+1, 0, 0]**
- step 里 `cos_supine = obs_g_x * (-1) = +1 * -1 = -0.95` ❌ 不触发

而 `roll_to_supine` reward 用的是 `ctx.gravity @ [-1,0,0]` = `[-1]·[-1,0,0]` = **+1** ✅ 触发。
两边用了不同源的 gravity,符号相反,导致 reward 能算对但 trigger 算错。

> 注意:`backend.get_gravity()` 返回的是**世界重力常量** [0,0,-9.81],
> 不是 per-env body-frame 投影重力,不能用于姿态判断。body-frame projected
> gravity 要走 `get_sensor_data(upvector)`。

### 修复
step() 里改用和 reward 同源的 upvector sensor:

```python
g_body = self._backend.get_sensor_data(self._cfg.sensor.upvector)  # (N,3) body-frame, 同 ctx.gravity
cos_supine = g_body[:, 0] * (-1.0)   # 仰卧 g_body=[-1,0,0] → cos=+1
```

### 教训
**trigger 逻辑和 reward 逻辑如果都依赖 gravity,必须用同一数据源。**
obs 里的 gravity 经过了 `-` 取反(很多 locomotion env 的惯例,把"重力方向"
转成"朝上方向"喂策略),直接拿来当 projected gravity 会符号反。
reward context 的 `ctx.gravity` 才是原始 sensor 值。任何状态门控/触发逻辑
应优先用 `ctx.gravity` 或 `backend.get_sensor_data(upvector)`,不要读 obs。

---

## 根因 2:frame 推进太快,ref 跑到站立帧时 robot 还躺着

### 现象
符号修复后 `_tracking_frame` 能触发了,但 `dof_tracking` 仍恒 0。
dry-run 发现:tracking_frame 触发后每步 +1,58 帧轨迹 ~58 步(1.16 秒)就推到末帧(站立姿态)。
而 robot 物理上不可能 1.16 秒从仰卧起身,于是:

- robot 还躺在地面(base_z≈0.12)
- 但 `_tracking_frame` 已递增到 50+(站立帧)
- ref_dof 是站立姿态的关节角
- cur_dof 是躺着姿态的关节角
- err 巨大 → `exp(-err/σ²)` ≈ 0

### 诊断
HumanUP 轨迹按 50fps 采集(58 帧 = 1.16 秒起身),但训练 env 是 150Hz,
robot 起身需要数秒。固定每步 frame+1 让 ref 远超 robot 实际进度。

### 修复
改成 **adaptive frame advance**:frame 不再按步数推进,而是按 robot 当前进度反查。

轨迹的 base_z 单调递增(0.120 仰卧 → 0.754 站立),用 robot 当前 base_z
找轨迹里 base_z 最接近的帧:

```python
# 预计算(轨迹加载时):每帧 base_z,从 head_height 映射
self._getup_traj_base_z = 0.12 + (0.754 - 0.12) * (hh - 0.054) / (1.278 - 0.054)

# step() 里:已触发的 env,frame 跟随 base_z
tracking = self._tracking_frame >= 0
cur_z = self._backend.get_base_pos()[:, 2]
ref_z = self._getup_traj_base_z  # (T,) 升序
idx = np.argmin(np.abs(cur_z[tracking, None] - ref_z[None, :]), axis=1)
self._tracking_frame[tracking] = idx
```

这样 ref 永远匹配 robot 当前高度:robot 贴地→frame 0(仰卧帧),
robot 半跪→frame ~29,robot 站立→frame 57。tracking err 才有意义。

**前提**:轨迹的参照量(这里是 base_z)必须随 frame 单调,否则 argmin 会跳帧。
HumanUP getup 轨迹 base_z 单调递增,满足。验证:`np.all(np.diff(base_z) >= -0.01)`。

### 教训
**轨迹 tracking 的相位推进必须和 robot 实际进度挂钩,不能按固定步数推进。**
否则 ref 跑在 robot 前面,err 必然巨大,tracking reward 恒 0。
motion_tracking env 里的 `sampling_mode="adaptive"` 解决的就是这个问题。
对非 motion_tracking env 自建 tracking 时,要选一个单调的进度量(高度/dof 范数/质心)
做 frame 反查。

---

## 根因 3:sigma 量级被 29 关节 sum-MSE 淹没

### 现象
前两个修复后 `dof_tracking` 仍极小(0.007 级),无学习梯度。
height_tracking 能涨(2.0+)但 dof_tracking 不动。

### 诊断
原实现用 **sum-MSE**:

```python
err = np.sum(np.square(cur_dof - ref_dof), axis=1)   # 29 关节平方误差之和
return np.exp(-err / (sigma * sigma))                # sigma=1.5
```

dry-run 实测噪声探索下 err ≈ 11(29 关节,每关节平均误差 ~0.6 rad / 35°):

- `exp(-11 / 1.5²) = exp(-4.9) = 0.0074`
- 乘权重 5 = 0.037,几乎无梯度

要 exp=0.3(有效信号),需 sigma² = 11/(-ln0.3) = 9.16 → sigma=3.0(单关节 172° 容忍,荒谬)。
sigma=1.5 期望单关节误差 < 0.18 rad(10°)才有有效信号,对早期探索太苛刻。

根因:**sum-MSE 让 sigma 失去物理意义**。29 关节累加后 err 量级是单关节的 ~29 倍,
sigma 要吸收这个倍数,无法用单关节角度解释。

### 修复
改成 **mean-MSE**(除以关节数),sigma 回到单关节物理量级:

```python
mse = np.sum(np.square(cur_dof - ref_dof), axis=1) / float(n)   # 平均到单关节
sigma = 0.5   # = 单关节 28° 容忍
return np.exp(-mse / (sigma * sigma))
```

验证修复后效果(同一噪声探索场景):

| 实现 | err/mse | exp | weighted(×5) |
|------|---------|-----|--------------|
| sum-MSE, σ=1.5(旧) | 11.0 | 0.007 | 0.037 ❌ |
| mean-MSE, σ=0.5(新) | 0.39 | 0.21 | 1.05 ✅ |

零 action 仰卧态(完美匹配):mse=0.16,exp=0.53,weighted=2.66。

sigma=0.5 的物理意义清晰:单关节平均误差 28° 以内开始有分,robot 越接近 ref 分越高。
早期探索(关节乱)仍有 0.2+ 的信号提供梯度,后期收紧可降到 σ=0.3。

### 教训
**多自由度 tracking reward 的 sigma 必须配合归一化方式定标。**
sum-MSE 时 sigma 要 ≈ √n × 单关节容忍,量级难调且无物理意义。
mean-MSE(或 RMS)让 sigma 直接对应单自由度误差,可解释、易调。
同样的陷阱:`penalty_orientation` 等用向量范数的 reward 也要注意是否归一化。

---

## 三修复的效果

逐个修复后 `reward/dof_tracking` 演变(每次只改一个变量):

| 阶段 | 修复 | dof_tracking |
|------|------|-------------|
| 起步 | 无 | 0.0000(恒) |
| 修根因1 | 符号(upvector) | 0.54(首值,但很快掉到 0.03) |
| 修根因2 | adaptive frame | 0.007(仍无梯度) |
| 修根因3 | mean-MSE σ=0.5 | **1.5+ 并上升** ✅ |

只有三个都修,dof_tracking 才真正提供学习信号。

---

## 调试方法论(供复用)

dof_tracking 恒 0 类问题的排查顺序:

1. **trigger 是否触发**:dry-run reset 后检查 `_tracking_frame` 是否从 -1 翻成 0。
   若没翻,查触发条件的符号/数据源(本例根因1)。
2. **frame 是否合理推进**:dry-run step N 次后看 frame 分布。若 frame 飙到末值
   而 robot 没动,说明推进速率和物理不匹配(本例根因2)。
3. **reward 量级是否有效**:dry-run 算 raw `exp(-err/σ²)`,看是否 >0.1。
   若 <0.01,查 err 量级和 sigma 定标是否匹配(本例根因3)。
4. **数据源一致性**:trigger / reward / obs 三处若都依赖某物理量(gravity/height/dof),
   确认用的是同一源,警惕 obs 的符号反转/归一化。

dry-run 脚本模板见本仓库 `/tmp/verify_*.py`(每次重启训练前先验证)。

---

## 后续:翘臀局部最优(未解决)

三修复后 dof_tracking 正常,但训到 iter 2600 评估发现 robot 学到了
**翘臀局部最优**:腿撑起下半身(腰/臀抬起),但头/上身还贴地,没真正站立。

原因:stand_feet(脚着地)+ height_exp/base_height(高度)+ dof_tracking(轨迹关节)
都能被"翘臀"姿态部分满足,而 uprightness_exp 权重不足以把它拉到真正直立。
这是 reward shaping 问题,下一步需加强上身抬起引导(如加 head/ torso 高度 reward、
或 uprightness 权重加大、或 base_height_target 对 fallen 更高)。

详见后续 Phase Q 设计。
