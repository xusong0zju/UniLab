"""GetUp Step 1: Static torque compensation — final verification (2026-06-29).

Key fix: PinocchioDynamicsModel(use_conjugate_quat=True) — eliminates 38.9 Nm
sign-reversal bug. g_pin = qbias_MJ (max|Δ|=0) for supine/prone/stand.

Results:
  - PD only:                      max = 25.4°
  - PD+g(q) FIXED:                max = 10.4° (2.4x improvement)
  - Remaining 10.4° = contact force (qfrc_constraint ≈ ±5 Nm) / kp
  - Contact forces CANNOT be fully predicted from Pinocchio (rigid vs soft contact)
  - Practical limit: ~10° with current kp, ~2° with 5x higher kp on low-gain joints

IMU: pelvis accelerometer working → base_supported=True for static supine.
"""
import os, sys, numpy as np, math
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
from unilab.control.pinocchio_model import PinocchioDynamicsModel
from unilab.control.actuator_switch import switch_to_motor_actuators


class IMUReader:
    def __init__(self, m):
        for i in range(m.nsensor):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) == "pelvis_acceleration":
                self._adr=m.sensor_adr[i]; self._dim=m.sensor_dim[i]; break
        else: raise RuntimeError("pelvis_acceleration not found")

    def read(self, d):
        al=d.sensordata[self._adr:self._adr+self._dim]
        R=_rotmat(d.qpos[3:7]); aw=R@al; an=aw+[0,0,-9.81]
        return {"accel_local":al,"accel_world":aw,"accel_net_z":an[2],"base_supported":abs(an[2])<4.9}


def _rotmat(q): w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def main():
    MODEL="src/unilab/assets/robots/g1/scene_flat.xml"
    m=mujoco.MjModel.from_xml_path(MODEL); d=mujoco.MjData(m)
    pm=PinocchioDynamicsModel(m, use_conjugate_quat=True)  # ← FIXED
    ai=switch_to_motor_actuators(m); kp,kd,fl,fu=ai.kp.copy(),ai.kd.copy(),ai.force_lower.copy(),ai.force_upper.copy()
    imu=IMUReader(m); stand=m.keyframe("stand").qpos.copy()
    jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,m.dof_jntid[i])or f"d{i}"for i in range(6,m.nv)]
    hh=math.radians(90)/2

    tgt_pos=stand.copy(); tgt_pos[2]=0.08; tgt_pos[3:7]=[math.cos(hh),0,-math.sin(hh),0]
    tgt=np.zeros(29); tgt[0]=-0.4;tgt[6]=-0.4;tgt[3]=0.9;tgt[9]=0.9
    tgt[14]=0.2;tgt[15]=0.67;tgt[22]=0.67;tgt[16]=0.;tgt[23]=0.
    tgt[18]=0.5;tgt[25]=0.5; tgt_pos[7:36]=tgt

    def run(label, tau_fn, n=2000):
        d.qpos[:]=tgt_pos; d.qvel[:]=0
        for _ in range(n): tau=tau_fn(d); d.ctrl[:]=np.clip(tau,fl,fu); mujoco.mj_step(m,d)
        err=tgt-d.qpos[7:36]
        print(f"  {label:30s}: max={np.degrees(np.abs(err).max()):5.1f}° mean={np.degrees(np.abs(err).mean()):4.1f}°")
        return err

    print("="*70)
    print("GetUp Step 1: FIXED Pinocchio Model (use_conjugate_quat=True)")
    print("="*70)

    e1=run("PD only", lambda d: kp*(tgt-d.qpos[7:36])-kd*d.qvel[6:35])
    e2=run("PD+g(q) FIXED", lambda d: (kp*(tgt-d.qpos[7:36])-kd*d.qvel[6:35]+pm.gravity(d.qpos[None,:],d.qvel[None,:])[0]))

    # Verify model
    mujoco.mj_forward(m,d)
    g_pin=pm.gravity(d.qpos[None,:],d.qvel[None,:])[0]
    qbias=d.qfrc_bias[6:35]; qc=d.qfrc_constraint[6:35]
    print(f"\nModel: max|g_pin-qbias|={np.abs(g_pin-qbias).max():.1f}Nm (was 38.9Nm with old model)")
    print(f"Contact: qfrc_constraint=[{qc.min():.1f},{qc.max():.1f}]Nm")
    print(f"Theoretical limit: max err = max|qc/kp| = {np.degrees(np.abs(qc/kp).max()):.1f}°")

    # IMU
    imud=imu.read(d)
    print(f"\nIMU: a_net_z={imud['accel_net_z']:.2f}, supported={imud['base_supported']}")

    # Key joints
    print(f"\nKey joints (°):")
    for idx in [0,3,14,15,18]:
        print(f"  {jn[idx]:25s}: PD={np.degrees(e1[idx]):+5.1f} +g(q)={np.degrees(e2[idx]):+5.1f} (kp={kp[idx]:.0f},qc={qc[idx]:+.1f}Nm)")

    print(f"\nResult: PD+g(q)FIXED = {np.degrees(np.abs(e2).max()):.1f}° max error (contact forces only)")
    print(f"        Improvement = {np.abs(e1).max()/max(np.abs(e2).max(),1e-6):.1f}x over PD-only")

    # Render
    os.makedirs("logs/getup_keyframes",exist_ok=True)
    import imageio; r=mujoco.Renderer(m,480,640)
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m,cam)
    cam.distance=2.5;cam.azimuth=90;cam.elevation=-25
    cam.lookat[:]=[d.qpos[0],d.qpos[1],d.qpos[2]+0.3]
    r.update_scene(d,camera=cam)
    imageio.imwrite("logs/getup_keyframes/stage1_fixed.png",r.render())
    print(f"Rendered: logs/getup_keyframes/stage1_fixed.png")
    return locals()


if __name__=="__main__":
    main()
