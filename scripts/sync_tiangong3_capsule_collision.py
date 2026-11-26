#!/usr/bin/env python3
"""Regenerate Retarget_X (and optionally DTMR) Tiangong3 XMLs using MLAB4_Fast tiangong3_capsule_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from retarget_x.tiangong3_capsule_collision import (
    TIANGONG3_CAPSULE_PROFILE_VERSION,
    audit_tiangong3_capsule_scene,
    build_tiangong3_mjx_robot_tree,
    build_tiangong3_mjx_scene_xml,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Rebuild Retarget_X Tiangong3 headless collision XMLs with MLAB4_Fast "
            f"{TIANGONG3_CAPSULE_PROFILE_VERSION} (39 analytic capsules/spheres, nmesh=0)."
        )
    )
    parser.add_argument(
        "--source-xml",
        type=Path,
        default=REPO_ROOT / "assets" / "tiangong3" / "tiangong3_mjx_torque_headless.xml",
        help="Source Tiangong3 robot XML (default: existing Retarget_X headless robot XML).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "tiangong3",
        help="Output directory for Retarget_X headless XMLs.",
    )
    parser.add_argument(
        "--dtmr-dir",
        type=Path,
        default=None,
        help="Optional DTMR repository root to also sync src/dtmr/models/tiangong3/ (headless + visual).",
    )
    parser.add_argument(
        "--cad-source-xml",
        type=Path,
        default=None,
        help="Optional full sdk_a_scene.xml used to also build visual tiangong3_mjx_torque.xml in --dtmr-dir.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_xml = out_dir / "tiangong3_mjx_torque_headless.xml"
    scene_xml = out_dir / "scene_mjx_torque_tracking_headless.xml"

    tree = build_tiangong3_mjx_robot_tree(args.source_xml, headless=True)
    tree.write(robot_xml, encoding="utf-8", xml_declaration=True)
    scene_xml.write_text(
        build_tiangong3_mjx_scene_xml("tiangong3_mjx_torque_headless.xml"),
        encoding="utf-8",
    )

    report = {"retarget_x_headless": audit_tiangong3_capsule_scene(scene_xml)}

    if args.dtmr_dir is not None:
        dtmr_models = args.dtmr_dir / "src" / "dtmr" / "models" / "tiangong3"
        dtmr_models.mkdir(parents=True, exist_ok=True)
        # Headless copy
        dtmr_robot_hl = dtmr_models / "tiangong3_mjx_torque_headless.xml"
        dtmr_scene_hl = dtmr_models / "scene_mjx_torque_tracking_headless.xml"
        tree.write(dtmr_robot_hl, encoding="utf-8", xml_declaration=True)
        dtmr_scene_hl.write_text(
            build_tiangong3_mjx_scene_xml("tiangong3_mjx_torque_headless.xml"),
            encoding="utf-8",
        )
        report["dtmr_headless"] = audit_tiangong3_capsule_scene(dtmr_scene_hl)

        # Visual copy (if sdk_a_scene.xml or existing visual xml is available)
        vis_src = args.cad_source_xml
        if vis_src is None and (dtmr_models / "tiangong3_mjx_torque.xml").exists():
            vis_src = dtmr_models / "tiangong3_mjx_torque.xml"
        if vis_src is not None and vis_src.exists():
            vis_tree = build_tiangong3_mjx_robot_tree(vis_src, headless=False)
            dtmr_robot_vis = dtmr_models / "tiangong3_mjx_torque.xml"
            dtmr_scene_vis = dtmr_models / "scene_mjx_torque_tracking.xml"
            vis_tree.write(dtmr_robot_vis, encoding="utf-8", xml_declaration=True)
            dtmr_scene_vis.write_text(
                build_tiangong3_mjx_scene_xml("tiangong3_mjx_torque.xml"),
                encoding="utf-8",
            )
            report["dtmr_visual"] = audit_tiangong3_capsule_scene(dtmr_scene_vis)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
