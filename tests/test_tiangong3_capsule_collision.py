"""Unit and CAD-parity tests for MLAB4_Fast tiangong3_capsule_v1 collision optimization."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from retarget_x.config import load_robot_spec
from retarget_x.tiangong3_capsule_collision import (
    TIANGONG3_CAPSULE_ALL_EXCLUDES,
    TIANGONG3_CAPSULE_FOOT_GEOM_NAMES,
    TIANGONG3_CAPSULE_HAND_GEOM_NAMES,
    TIANGONG3_CAPSULE_PROFILE_VERSION,
    TIANGONG3_CAPSULE_SPECS,
    TIANGONG3_EXPLICIT_CONTACT_PAIRS,
    apply_tiangong3_capsule_collisions,
    audit_tiangong3_capsule_scene,
)

CAD_SCENE_PATH = Path(os.environ.get("TIANGONG3_CAD_SCENE_XML", ""))
MLAB4_FAST_SRC = Path(os.environ.get("MLAB4_FAST_SRC", ""))


def test_tiangong3_spec_uses_capsule_v1_profile():
    spec = load_robot_spec("tiangong3")
    assert spec.collision_profile == TIANGONG3_CAPSULE_PROFILE_VERSION
    assert spec.foot_collision_geoms == TIANGONG3_CAPSULE_FOOT_GEOM_NAMES
    assert len(spec.foot_collision_geoms) == 8


def test_tiangong3_headless_scene_zero_mesh_39_primitives_and_clean_keyframes():
    spec = load_robot_spec("tiangong3")
    audit = audit_tiangong3_capsule_scene(spec.headless_scene_path)

    assert audit["profile_version"] == TIANGONG3_CAPSULE_PROFILE_VERSION
    assert (audit["nq"], audit["nv"], audit["nu"]) == (36, 35, 29)
    assert audit["n_collision_geoms"] == 39
    assert audit["n_mesh_collision_geoms"] == 0
    assert audit["nexclude"] == len(TIANGONG3_CAPSULE_ALL_EXCLUDES) == 20
    assert audit["npair"] == len(TIANGONG3_EXPLICIT_CONTACT_PAIRS)

    # Sub-0.1mm standing & crouched rocker sole accuracy, zero false self-contacts
    assert abs(audit["keyframes"]["home"]["foot_min_dist_mm"]) < 0.05
    assert audit["keyframes"]["home"]["n_self_contacts"] == 0
    assert abs(audit["keyframes"]["knees_bent"]["foot_min_dist_mm"]) < 0.1
    assert audit["keyframes"]["knees_bent"]["n_self_contacts"] == 0


def test_tiangong3_all_39_geoms_have_floor_condim3_pairs_and_no_excluded_self_pairs():
    spec = load_robot_spec("tiangong3")
    model = mujoco.MjModel.from_xml_path(str(spec.headless_scene_path))
    floor_id = model.geom("floor").id

    floor_paired_geoms: set[str] = set()
    excluded_bodies = {tuple(sorted(pair)) for pair in TIANGONG3_CAPSULE_ALL_EXCLUDES}
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if floor_id in (g1, g2):
            other = g2 if g1 == floor_id else g1
            floor_paired_geoms.add(model.geom(other).name)
            assert int(model.pair_dim[p]) == 3
        else:
            b1 = model.body(model.geom_bodyid[g1]).name
            b2 = model.body(model.geom_bodyid[g2]).name
            assert b1 != b2
            assert tuple(sorted((b1, b2))) not in excluded_bodies
            assert int(model.pair_dim[p]) == 1

    expected_geoms = {s.name for s in TIANGONG3_CAPSULE_SPECS}
    assert floor_paired_geoms == expected_geoms
    for hg in TIANGONG3_CAPSULE_HAND_GEOM_NAMES:
        assert hg in floor_paired_geoms


def test_apply_tiangong3_capsule_collisions_on_mjspec():
    spec = load_robot_spec("tiangong3")
    mjspec = mujoco.MjSpec.from_file(str(spec.headless_scene_path))
    info = apply_tiangong3_capsule_collisions(mjspec)
    compiled = mjspec.compile()
    assert info["version"] == TIANGONG3_CAPSULE_PROFILE_VERSION
    assert info["mesh_collision_geoms"] == 0
    assert info["analytic_collision_geoms"] == 39
    assert tuple(info["foot_collision_geoms"]) == TIANGONG3_CAPSULE_FOOT_GEOM_NAMES
    assert compiled.nmesh == 0
    assert compiled.ngeom == 40
    assert compiled.nexclude == 20


@pytest.mark.skipif(
    not (str(CAD_SCENE_PATH) and CAD_SCENE_PATH.is_file()),
    reason="Set TIANGONG3_CAD_SCENE_XML to run CAD parity check",
)
def test_tiangong3_capsule_v1_matches_cad_dynamics_and_rocker_sole():
    spec = load_robot_spec("tiangong3")
    m_cad = mujoco.MjModel.from_xml_path(str(CAD_SCENE_PATH))
    m_rx = mujoco.MjModel.from_xml_path(str(spec.headless_scene_path))

    # Bit-exact dynamics & actuator contract vs CAD
    assert np.allclose(m_rx.body_mass, m_cad.body_mass, atol=1e-9)
    assert np.allclose(m_rx.body_inertia, m_cad.body_inertia, atol=1e-9)
    assert np.allclose(m_rx.jnt_range[1:], m_cad.jnt_range[1:], atol=1e-9)
    assert np.allclose(m_rx.dof_armature, m_cad.dof_armature, atol=1e-9)
    assert np.allclose(m_rx.dof_damping, m_cad.dof_damping, atol=1e-9)
    assert np.allclose(m_rx.actuator_ctrlrange, m_cad.actuator_ctrlrange, atol=1e-9)

    # Plantarflexed/dorsiflexed 1.35-deg rocker sole accuracy vs CAD (< 0.7 mm across ankle pitches)
    d_cad, d_rx = mujoco.MjData(m_cad), mujoco.MjData(m_rx)
    f_cad, f_rx = m_cad.geom("floor").id, m_rx.geom("floor").id
    cad_feet = [m_cad.geom("ankle_roll_l_link_collision").id, m_cad.geom("ankle_roll_r_link_collision").id]
    rx_feet = [m_rx.geom(g).id for g in TIANGONG3_CAPSULE_FOOT_GEOM_NAMES]

    for pitch in (-0.35, -0.15, 0.0, 0.15, 0.35):
        mujoco.mj_resetDataKeyframe(m_rx, d_rx, 0)
        q = d_rx.qpos.copy()
        q[11] = pitch
        q[17] = pitch
        d_cad.qpos[:] = q
        d_rx.qpos[:] = q
        mujoco.mj_forward(m_cad, d_cad)
        mujoco.mj_forward(m_rx, d_rx)
        dc = min(mujoco.mj_geomDistance(m_cad, d_cad, f_cad, g, 3.0, None) for g in cad_feet) * 1000.0
        dr = min(mujoco.mj_geomDistance(m_rx, d_rx, f_rx, g, 3.0, None) for g in rx_feet) * 1000.0
        assert abs(dr - dc) < 0.7, f"Ankle pitch {pitch}: CAD={dc:.3f}mm vs capsule_v1={dr:.3f}mm"


@pytest.mark.skipif(
    not (
        str(CAD_SCENE_PATH)
        and CAD_SCENE_PATH.is_file()
        and str(MLAB4_FAST_SRC)
        and (MLAB4_FAST_SRC / "mjlab/utils/tiangong_capsule_collision.py").is_file()
    ),
    reason="Set TIANGONG3_CAD_SCENE_XML and MLAB4_FAST_SRC to run MLAB4_Fast parity check",
)
def test_tiangong3_capsule_v1_exact_parity_with_mlab4_fast():
    if str(MLAB4_FAST_SRC) not in sys.path:
        sys.path.insert(0, str(MLAB4_FAST_SRC))
    from mjlab.utils.tiangong_capsule_collision import (  # type: ignore[import-not-found]
        TIANGONG3_CAPSULE_PROFILE_VERSION as MLAB_VERSION,
        apply_tiangong3_capsule_collisions as mlab_apply,
    )

    assert MLAB_VERSION == TIANGONG3_CAPSULE_PROFILE_VERSION
    spec_mlab = mujoco.MjSpec.from_file(str(CAD_SCENE_PATH))
    mlab_apply(spec_mlab)
    m_mlab = spec_mlab.compile()

    rx_spec = load_robot_spec("tiangong3")
    m_rx = mujoco.MjModel.from_xml_path(str(rx_spec.headless_scene_path))

    rx_geoms = {m_rx.geom(i).name: i for i in range(m_rx.ngeom) if m_rx.geom(i).name != "floor"}
    mlab_geoms = {m_mlab.geom(i).name: i for i in range(m_mlab.ngeom) if m_mlab.geom(i).name != "floor"}
    assert set(rx_geoms) == set(mlab_geoms)
    for gname, rx_id in rx_geoms.items():
        ml_id = mlab_geoms[gname]
        assert m_rx.geom_type[rx_id] == m_mlab.geom_type[ml_id]
        assert m_rx.body(m_rx.geom_bodyid[rx_id]).name == m_mlab.body(m_mlab.geom_bodyid[ml_id]).name
        assert np.allclose(m_rx.geom_size[rx_id], m_mlab.geom_size[ml_id], atol=1e-6)
        assert np.allclose(m_rx.geom_pos[rx_id], m_mlab.geom_pos[ml_id], atol=1e-6)
        q1, q2 = m_rx.geom_quat[rx_id], m_mlab.geom_quat[ml_id]
        assert np.allclose(q1, q2, atol=1e-5) or np.allclose(q1, -q2, atol=1e-5)
