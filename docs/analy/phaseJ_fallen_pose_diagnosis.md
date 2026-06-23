# Phase J 起身训练诊断 — fallen 姿态"半跪/微倾"非真倒地

> 验证日期:2026-06-23
> 对象:Phase J model_6000(aux 衰减方案,stand_feet 4.85)

## 1. 训练成果(属实)

- aux 衰减(step 500-5000 线性 1.0→0)成功:aux=0 后 stand_feet 保持 4.85-4.90
- iter 4500 后收敛(stand_feet CV=0.006)
- **半跪/低位倾斜起身确实学会**(stand_feet 4.85 ≈ 97% 站住)

## 2. 关键发现:fallen 姿态"倾斜被物理抹平"

实测(phaseJ env, 禁辅助):
- reset 后 fallen base_z = **0.486**(设的 0.40,因倾斜略高)
- step1 后 base_z = 0.486(未被 reset,terminated 仅 7%)
- **tilt = 31°**(设的 80° 倾斜,实际只剩 31°!)

**根因**:build_reset_plan 设 fallen 倾斜四元数(80°/70°/45°),但 MuJoCo 物理 settle(reset 后 mj_forward)把机器人姿态"拉直"——倾斜被接触力/重力抹平到 31°(接近直立)。

**结果**:fallen env 实际是"半跪/微倾"(base_z 0.486, tilt 31°),**不是真倒地**。stand_feet 4.85 是在这种接近直立的姿态上训出来的。

## 3. 真平躺验证失败(set_state 被 autoreset)

手动 set_state 构造真平躺(base_z=0.20 + tilt 170°):
- set_state 后 base_z=0.20 ✅(生效)
- **step1 后 base_z 变 0.55**(terminated=True 触发 autoreset,reset 回 fallen 0.40/0.55)
- 原因:170° 倾斜 > max_tilt=80°,立即终止→autoreset

**结论**:model_6000 没见过真平躺,且当前终止条件(max_tilt=80°)不允许真平躺训练。

## 4. 当前成果定位(诚实)

| 能力 | 是否学会 | 依据 |
|---|---|---|
| 半跪起身(base_z 0.486, tilt 31°) | ✅ | stand_feet 4.85, aux=0 自主 |
| 低位倾斜起身(tilt 31-50°) | ✅ | 同上 |
| 真平躺起身(base_z 0.20, tilt 170°) | ❌ 未训 | 姿态被物理抹平+终止不允许 |
| 仰卧/俯卧/侧卧真倒地 | ❌ 未训 | 倾斜设 80° 实际 31° |

## 5. 若要真倒地起身,需解决两个问题

1. **倾斜被抹平**:设 80° 实际 31°。需让倾斜"设进去后不被物理拉直"——可能要禁 reset 后的 mj_forward settle,或用更高 base_z 让倾斜可保持
2. **终止条件**:max_tilt=80° 不允许 170° 平躺。需对 fallen env 单独放宽 max_tilt 到 180°
3. **关节姿态**:当前用 stand keyframe(腿伸直),真倒地腿应弯曲(用 kneeling 关节)
4. **G 阶段风险**:真平躺起身 MuJoCo 150Hz 失败 6 次,需 aux 衰减课程(已验证有效)+ 可能更长训练

## 6. 结论

Phase J model_6000 是"半跪/微倾起身"(stand_feet 4.85 属实),**不是真倒地起身**。若目标仅是"被推倒后能半跪爬起",当前成果够用。若要"真平躺仰/俯/侧起身",需重训解决倾斜抹平+终止+关节姿态三问题,有 G 阶段失败风险。

## 7. 视频+数据复核(2026-06-23 16:50)

**用户看视频反馈**:开头大量站着走着,只有极个别起身。

**数据核实**(phaseJ env, 256 envs 全 fallen, 禁 aux, reset 后第 0 步):
| 姿态 | 占比 |
|---|---|
| z<0.3(真倒地) | **0%** |
| 0.3≤z<0.5(半跪/低位) | 45% |
| z≥0.5(接近站立) | **55%** |
| tilt<30°(接近直立) | **55%** |
| tilt>60°(明显倾斜) | 28% |

**证实**:reset 后 **55% fallen env base_z≥0.5 且 tilt<30°——基本站着/微倾,没倒下**。stand_feet 4.85 有水分:大部分 env 本来就接近站立,"起身"对它们无意义。真正从倒下起步的仅 ~28%(tilt>60°)。

**根因**:
1. 半跪 keyframe(50% env)base_z=0.55 接近站立
2. 倾斜设 80° 被物理抹平到 30°(拉直)
3. 只有 28% tilt>60° 的 env 真倒着

**结论**:当前训练非真起身训练,是"站着/微倾→保持站"。要真起身必须让 env 真倒下(base_z<0.3 + tilt 大),需重训解决:
- 倾斜抹平(禁 reset 后 mj_forward settle,或更高 base_z 让倾斜可保持)
- 终止条件(max_tilt 对 fallen 放宽到 180°)
- 关节姿态(用弯曲关节非 stand keyframe)
- G 阶段风险:真平躺起身 MuJoCo 150Hz 失败 6 次,需 aux 衰减课程(已验证有效)+ 可能更长训练
