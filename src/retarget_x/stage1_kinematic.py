"""Stage 1: Enhanced Kinematic Retargeting via hhtools (source_plane scaling + multi-pose calibration)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
import time
import types

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import RobotSpec

CALIBRATION_POSES = [
    ("T90", 90.0, 0.0),
    ("T60", 60.0, 0.0),
    ("T90_elbow45", 90.0, 45.0),
]


@dataclass
class KinematicResult:
    qpos: np.ndarray          # (T, 7 + dof) root_pos(3) | root_quat_wxyz(4) | joints(dof)
    qvel: np.ndarray          # (T, 6 + dof)
    fps: float
    calibration_name: str
    seconds: float
    selection_score: float


def load_soma_g1_csv_resampled(csv_path: Path, target_fps: float = 50.0, source_fps: float = 120.0):
    """Parse a 120Hz BONES-SEED / SOMA G1 CSV and resample to target_fps."""
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    header = rows[0]
    vals = np.asarray(rows[1:], dtype=np.float64)
    col = {h: i for i, h in enumerate(header)}
    pos = vals[:, [col["root_translateX"], col["root_translateY"], col["root_translateZ"]]] * 0.01
    euler = vals[:, [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]]]
    quat_xyzw = Rotation.from_euler("xyz", euler, degrees=True).as_quat()

    root_cols = {
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
    }
    jcols = [h for h in header if h not in root_cols]
    jnames = [h[:-4] if h.endswith("_dof") else h for h in jcols]
    jpos = np.deg2rad(vals[:, [col[h] for h in jcols]])

    n_src = len(pos)
    dur = (n_src - 1) / source_fps
    n_dst = max(2, int(round(dur * target_fps)) + 1)
    t_src = np.linspace(0.0, dur, n_src)
    t_dst = np.linspace(0.0, dur, n_dst)

    pos_out = np.column_stack([np.interp(t_dst, t_src, pos[:, i]) for i in range(3)])
    jpos_out = np.column_stack([np.interp(t_dst, t_src, jpos[:, i]) for i in range(jpos.shape[1])])

    q_cont = quat_xyzw.copy()
    for i in range(1, len(q_cont)):
        if np.dot(q_cont[i], q_cont[i - 1]) < 0:
            q_cont[i] = -q_cont[i]
    q_out = np.column_stack([np.interp(t_dst, t_src, q_cont[:, i]) for i in range(4)])
    q_out /= np.linalg.norm(q_out, axis=1, keepdims=True)
    return pos_out, q_out, jpos_out, jnames


def _score_candidate(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    """Fast engineering score combining floor penetration, self-collision, and joint step smoothness."""
    d = mujoco.MjData(model)
    floor_id = model.geom("floor").id
    fp_list, sp_list = [], []
    for row in qpos:
        d.qpos[:] = row
        mujoco.mj_forward(model, d)
        fp, sp = 0.0, 0.0
        for c in d.contact[: d.ncon]:
            if floor_id in (c.geom1, c.geom2):
                fp = max(fp, -float(c.dist))
            else:
                sp = max(sp, -float(c.dist))
        fp_list.append(fp)
        sp_list.append(sp)
    dq = np.diff(qpos[:, 7:], axis=0)
    return float(5.0 * np.quantile(fp_list, 0.95) + 5.0 * np.quantile(sp_list, 0.95) + 0.25 * np.quantile(np.abs(dq), 0.99))


def retarget_g1_csv_with_hhtools(
    csv_path: Path,
    robot_spec: RobotSpec,
    source_urdf: Path,
    target_urdf: Path,
    work_dir: Path,
    *,
    multipose: bool = False,
    ik_iterations: int = 48,
    target_fps: float = 50.0,
) -> KinematicResult:
    """Run hhtools NewtonBasicPipeline with exact source_plane height ratio and optional multi-pose selection."""
    from hhtools.core.grounding import SOURCE_FLOOR_META_KEY
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.robot_to_robot import _build_scaler_config, source_trajectory_to_motion
    from hhtools.robot.base import RobotPreset
    from hhtools.robot.loader import load_robot
    from hhtools.robot.retarget_profile import build_feet_stabilizer_config, build_pipeline_config_for_preset

    pos_50, quat_xyzw_50, jpos_50, gnames = load_soma_g1_csv_resampled(csv_path, target_fps=target_fps)
    q_g1 = np.concatenate([pos_50, quat_xyzw_50, jpos_50], axis=1).astype(np.float32)

    gmap = {"hips": "pelvis", "chest": "torso_link"}
    tmap = {"hips": "pelvis", "chest": "waist_pitch_link"}
    for side, suffix in [("left", "l"), ("right", "r")]:
        for k, a, b in [
            ("shoulder", "shoulder_roll", "shoulder_roll"),
            ("elbow", "elbow", "elbow_pitch"),
            ("wrist", "wrist_yaw", "wrist_roll"),
            ("hip", "hip_roll", "hip_roll"),
            ("knee", "knee", "knee_pitch"),
            ("ankle", "ankle_roll", "ankle_roll"),
        ]:
            gmap[f"{side}_{k}"] = f"{side}_{a}_link"
            tmap[f"{side}_{k}"] = f"{b}_{suffix}_link"

    weights = {
        "t_weight": {k: (10.0 if k == "hips" else 2.0 if "ankle" in k else 1.0) for k in tmap},
        "r_weight": {k: (2.0 if k == "hips" else 0.2 if any(s in k for s in ["hip", "shoulder"]) else 1.0) for k in tmap},
    }
    src = load_robot(RobotPreset("g1_pair", "G1", work_dir, source_urdf, ik_map=gmap, dof_order=tuple(gnames)))
    tgt = load_robot(
        RobotPreset(
            "tiangong_pair",
            robot_spec.name,
            work_dir,
            target_urdf,
            ik_map=tmap,
            weights=weights,
            dof_order=robot_spec.joint_names,
            meta={"retarget": dict(ground_collision_weight=10.0, ground_collision_z=0.0,
                                   foot_clamp_anti_penetration=False, clip_floor_snap=False)},
        )
    )
    model = mujoco.MjModel.from_xml_path(str(robot_spec.headless_scene_path))
    poses = CALIBRATION_POSES if multipose else [CALIBRATION_POSES[0]]
    gzero, tzero = src.zero_configuration(), tgt.zero_configuration()

    best: KinematicResult | None = None
    for cname, spread, bend in poses:
        gp, tp = dict(gzero), dict(tzero)
        for side, suffix, sign in [("left", "l", 1), ("right", "r", -1)]:
            gp[f"{side}_shoulder_roll_joint"] = sign * np.deg2rad(spread)
            gp[f"{side}_elbow_joint"] = np.pi / 2 - np.deg2rad(bend)
            tp[f"shoulder_roll_{suffix}_joint"] = sign * np.deg2rad(spread)
            tp[f"elbow_pitch_{suffix}_joint"] = -np.deg2rad(bend)
        src.zero_configuration = types.MethodType(lambda self, pose=gp: dict(pose), src)
        cfg, ref = _build_scaler_config(src, tgt, tp)

        # Exact SDK sole-height ratio (eliminates fallback 1.3m stature bug)
        ratio = robot_spec.height_ratio_vs_g1
        scales = dict(cfg.joint_scales)
        scales[cfg.root_joint] = ratio
        cfg = replace(cfg, joint_scales=scales, root_z_offset=0.0,
                      robot_pelvis_height=None, root_trajectory_scale=ratio)

        pipeline = NewtonBasicPipeline(
            tgt,
            scaler_config=cfg,
            pipeline_config=build_pipeline_config_for_preset(tgt.preset, "robot_g1_pair", ik_iterations=ik_iterations),
            feet_stabilizer_config=build_feet_stabilizer_config(tgt.preset, "robot_g1_pair", model=tgt),
            human_height=float(ref.height_m),
            source_to_canonical={n: n for n in ref.joint_names},
        )
        motion = source_trajectory_to_motion(src, q_g1, tuple(gnames), framerate=target_fps, name=csv_path.stem)
        motion.meta[SOURCE_FLOOR_META_KEY] = 0.0

        t0 = time.monotonic()
        ret = pipeline.run(motion)
        elapsed = time.monotonic() - t0

        q = np.zeros((len(ret.joint_q), model.nq), dtype=np.float64)
        q[:, 0:3] = ret.joint_q[:, 0:3]
        q[:, 3:7] = ret.joint_q[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        idx_map = [7 + ret.dof_names.index(j) for j in robot_spec.joint_names]
        q[:, 7:] = ret.joint_q[:, idx_map]

        qd = np.zeros((len(q), model.nv), dtype=np.float64)
        for i in range(len(q)):
            lo, hi = max(0, i - 1), min(len(q) - 1, i + 1)
            mujoco.mj_differentiatePos(model, qd[i], (hi - lo) / target_fps, q[lo], q[hi])

        score = _score_candidate(model, q)
        cand = KinematicResult(qpos=q, qvel=qd, fps=target_fps, calibration_name=cname, seconds=elapsed, selection_score=score)
        if best is None or cand.selection_score < best.selection_score:
            best = cand

    assert best is not None
    return best
