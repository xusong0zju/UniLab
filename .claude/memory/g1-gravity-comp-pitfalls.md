---
name: g1-gravity-comp-pitfalls
description: Key pitfalls encountered during G1 gravity compensation implementation
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8bd6825e-9f4d-4a51-b690-2b467fbb06bf
---

1. **G1 ctrlrange=[0,0]**：力矩限制在 `actuator_forcerange` 中，不是 ctrlrange。切换 motor actuator 后必须设 ctrlrange=forcerange。
2. **gymnasium float32 overflow**：motor actuator 的 ctrlrange=forcerange（如[-88,88]）导致 gymnasium Box 溢出。解决方案：`_init_action_space()` 返回 [-1,1]^29，policy 输出目标角而非力矩。
3. **继承 G1WalkEnv 会创建重复 backend**：G1WalkEnv.__init__ 内部创建自己的 backend。G1WalkFlatGC 必须继承 G1BaseEnv，手动复制 G1WalkEnv 的设置代码。
4. **G1BaseEnv 无 _np_dtype**：需用 `get_global_dtype()` 替代。
5. **DR validation 访问 control_config.Kp**：G1WalkGCControlConfig 没有 Kp 属性。需自定义 DR Provider，override validate() 和 build_reset_plan()。
6. **Pinocchio 四元数约定**：MuJoCo (w,x,y,z) ↔ Pinocchio (x,y,z,w)，_align_state 中必须转换。

**Why:** 这些坑耗时较多，避免下次重蹈覆辙。
**How to apply:** 涉及 MuJoCo actuator 切换、env 继承、Pinocchio 集成时逐一检查。
