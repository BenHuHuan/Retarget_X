"""Robot specification and pipeline configuration loader for Retarget_X."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RobotSpec:
    name: str
    dof: int
    standing_root_height_m: float
    g1_standing_root_height_m: float
    mass_kg: float
    headless_scene_path: Path
    dtmr_robot_name: str
    joint_names: tuple[str, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    foot_collision_geoms: tuple[str, ...]
    conditioning: dict[str, Any]
    mpc_profiles: dict[str, dict[str, Any]]

    @property
    def height_ratio_vs_g1(self) -> float:
        return self.standing_root_height_m / self.g1_standing_root_height_m


def load_robot_spec(name_or_path: str | Path) -> RobotSpec:
    p = Path(name_or_path)
    if not p.exists():
        candidate = REPO_ROOT / "configs" / "robots" / f"{name_or_path}.yaml"
        if candidate.exists():
            p = candidate
        else:
            raise FileNotFoundError(f"Robot config not found: {name_or_path} (tried {candidate})")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    scene_rel = Path(raw["model"]["headless_scene"])
    scene_path = scene_rel if scene_rel.is_absolute() else (REPO_ROOT / scene_rel)

    return RobotSpec(
        name=str(raw["name"]),
        dof=int(raw["dof"]),
        standing_root_height_m=float(raw["standing_root_height_m"]),
        g1_standing_root_height_m=float(raw.get("g1_standing_root_height_m", 0.79386)),
        mass_kg=float(raw["mass_kg"]),
        headless_scene_path=scene_path,
        dtmr_robot_name=str(raw["model"]["dtmr_robot_name"]),
        joint_names=tuple(raw["joint_names"]),
        kp=tuple(float(x) for x in raw["pd"]["kp"]),
        kd=tuple(float(x) for x in raw["pd"]["kd"]),
        foot_collision_geoms=tuple(raw["foot_collision_geoms"]),
        conditioning=dict(raw.get("conditioning", {})),
        mpc_profiles=dict(raw.get("mpc_profiles", {})),
    )
