"""把HumanUP getup轨迹(29DoF对齐)转成UniLab motion_loader NPZ格式.
每帧灌qpos进MuJoCo, 取14个tracked body的pos/quat, 差分算速度."""
import os, sys, pickle, numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import mujoco

ALIGNED = "logs/g1_reference_traj/getup_aligned29.npz"
OUT = "logs/g1_reference_traj/getup_motion.npz"
MODEL = "src/unilab/assets/robots/g1/scene_flat.xml"
FPS = 50
# Store ALL bodies (by MuJoCo body id order) so motion_loader's body_indices
# (MuJoCo body ids) can directly index the body axis. Previously stored only 14
# tracked bodies, causing IndexError (id 16 > 14).

d = np.load(ALIGNED)
dof_pos = d["dof_pos"].astype(np.float32)  # (58, 29)
head_height = d["head_height"].astype(np.float32)  # (58,)
N = dof_pos.shape[0]
m = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(m)
stand = m.keyframe("stand").qpos.copy()  # (36,) base(7)+29joint

nb = m.nbody  # ALL bodies

# 每帧: base_z从head_height映射 + base quat绕y轴翻转(倒地tilt大, 站立0)
# 绕y翻90°不穿地(min_body_z 0.017), 绕x翻或identity穿地(-0.6)
def head_to_base_z(h):
    # 倒地head 0.054→base 0.12, 站立head 1.278→base 0.754
    return 0.12 + (0.754 - 0.12) * (h - 0.054) / (1.278 - 0.054)

def head_to_base_quat(h):
    # tilt: 倒地90°→站立0°, 绕body y轴
    tilt_deg = 90.0 * (1.278 - h) / (1.278 - 0.054)
    tilt_deg = max(0.0, min(90.0, tilt_deg))
    h_half = np.deg2rad(tilt_deg) / 2
    return np.array([np.cos(h_half), 0, np.sin(h_half), 0])  # 绕y轴

body_pos_w = np.zeros((N, nb, 3), dtype=np.float32)
body_quat_w = np.zeros((N, nb, 4), dtype=np.float32)
for i in range(N):
    qpos = stand.copy()
    qpos[2] = float(head_to_base_z(head_height[i]))
    qpos[3:7] = head_to_base_quat(head_height[i])  # tilt from head_height
    qpos[7:7+29] = dof_pos[i]
    data.qpos[:] = qpos
    mujoco.mj_forward(m, data)
    body_pos_w[i] = data.xpos[:nb]
    body_quat_w[i] = data.xquat[:nb]

# Clamp body z >= 0.02 (safety, 起身中间帧可能微穿)
body_pos_w[:, :, 2] = np.maximum(body_pos_w[:, :, 2], 0.02)

# 速度差分
dt = 1.0 / FPS
joint_vel = np.gradient(dof_pos, axis=0, edge_order=1) / dt  # (N,29)
body_lin_vel_w = np.gradient(body_pos_w, axis=0, edge_order=1) / dt  # (N,nb,3)
# ang_vel from quat diff (简化: 数值差分转角速度)
def quat_to_ang_vel(q, dt):
    # q: (N,nb,4), 返回 (N,nb,3) 数值差分
    # 简化: 用相邻quat相对旋转的axis*angle/dt
    out = np.zeros((q.shape[0], q.shape[1], 3), dtype=np.float32)
    for i in range(1, q.shape[0]):
        for b in range(q.shape[1]):
            # q_rel = q[i] * inv(q[i-1])
            q0 = q[i-1, b]; q1 = q[i, b]
            # inv(q0)
            q0inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
            # q_rel = q1 * q0inv (quat mul)
            w = q1[0]*q0inv[0] - q1[1]*q0inv[1] - q1[2]*q0inv[2] - q1[3]*q0inv[3]
            x = q1[0]*q0inv[1] + q1[1]*q0inv[0] + q1[2]*q0inv[3] - q1[3]*q0inv[2]
            y = q1[0]*q0inv[2] - q1[1]*q0inv[3] + q1[2]*q0inv[0] + q1[3]*q0inv[1]
            z = q1[0]*q0inv[3] + q1[1]*q0inv[2] - q1[2]*q0inv[1] + q1[3]*q0inv[0]
            qrel = np.array([w, x, y, z])
            angle = 2 * np.arccos(np.clip(abs(qrel[0]), 0, 1))
            axis = qrel[1:] / (np.linalg.norm(qrel[1:]) + 1e-8)
            out[i, b] = axis * angle / dt
    return out
body_ang_vel_w = quat_to_ang_vel(body_quat_w, dt)

np.savez(OUT, fps=FPS, joint_pos=dof_pos, joint_vel=joint_vel.astype(np.float32),
         body_pos_w=body_pos_w, body_quat_w=body_quat_w,
         body_lin_vel_w=body_lin_vel_w.astype(np.float32),
         body_ang_vel_w=body_ang_vel_w.astype(np.float32))
print(f"✅ 保存 {OUT}: {N}帧, {nb}bodies, joint_pos{dof_pos.shape}")
print(f"  body_pos_w: {body_pos_w.shape}, 首帧pelvis_z={body_pos_w[0,0,2]:.3f} 末帧={body_pos_w[-1,0,2]:.3f}")
