# Dataset converters

One file per source root, flat, prefixed by family. Run each from the repo root:
`pixi run python scripts/convert/<name>.py --help`. Shared helpers live in `_common.py`
(sibling imports are plain `from _common import ...`, which works because Python puts the
script's own folder on `sys.path`). Converters have no unit tests by rule; check one with
`--help`, `--dry-run` where it has one, and `--validate` / `scripts/dataset_tools/check_status.py`
on its output.

## Rename table (2026-10-19)

The converters write their own path into every session they produce, as `annotator_tool`
(calms21, johnson_mouse, johnson_mouse_aug, v4) or `[provenance].converter` (the rest). That string
is **provenance**: it names the script that built the root. Roots converted before the move carry
the OLD path in the left column; roots converted after it carry the NEW one. Both name the same
code. Do not rewrite an existing root's string to "fix" it -- regenerate the root instead.

| old path (in shipped roots) | new path |
|---|---|
| `scripts/convert_v4.py` | `scripts/convert/v4.py` |
| `scripts/convert_annotated.py` | `scripts/convert/annotated.py` |
| `scripts/convert_apt_lbl.py` | `scripts/convert/apt_lbl.py` |
| `scripts/convert_aptv2.py` | `scripts/convert/aptv2.py` |
| `scripts/convert_calms21.py` | `scripts/convert/calms21.py` |
| `scripts/convert_facemap.py` | `scripts/convert/facemap.py` |
| `scripts/convert_leap_fly.py` | `scripts/convert/leap_fly.py` |
| `scripts/convert_minimaze.py` | `scripts/convert/minimaze.py` |
| `scripts/convert_multifly.py` | `scripts/convert/multifly.py` |
| `scripts/convert_rodent3d.py` | `scripts/convert/rodent3d.py` |
| `scripts/convert_rp5k.py` | `scripts/convert/rp5k.py` |
| `scripts/convert_schulze.py` | `scripts/convert/schulze.py` |
| `scripts/convert_spaceanimal.py` | `scripts/convert/spaceanimal.py` |
| `scripts/convert_tuthill_fly.py` | `scripts/convert/tuthill_fly.py` |
| `scripts/convert_udarp.py` | `scripts/convert/udarp.py` |
| `scripts/convert_johnson.py` | `scripts/convert/johnson_mouse.py` |
| `scripts/convert_johnson_aug.py` | `scripts/convert/johnson_mouse_aug.py` |
| `scripts/convert_johnson_fly.py` | `scripts/convert/johnson_fly.py` |
| `scripts/deepfly3d_common.py` | `scripts/convert/deepfly3d_common.py` |
| `scripts/convert_deepfly3d_v3.py` | `scripts/convert/deepfly3d_v3.py` |
| `scripts/convert_qdmouse4m.py` | `scripts/convert/qdmouse4m.py` |
| `scripts/convert_qdmouse4m_fluo.py` | `scripts/convert/qdmouse4m_fluo.py` |
| `scripts/clean_qdmouse4m.py` | `scripts/convert/qdmouse4m_clean.py` |
| `scripts/add_qdmouse4m_fluo_cameras.py` | `scripts/convert/qdmouse4m_add_fluo_cameras.py` |

`scripts/convert_deepfly3d_v3.py` is now split across `deepfly3d_v3.py` (CLI, discovery,
conversion), `deepfly3d_h5.py` (HDF5 reading) and `deepfly3d_rig.py` (calibration rig).

## Other scripts moved in the same change

| old path | new path |
|---|---|
| `scripts/convert_deepfly3d_v4_posetail.py` | `scripts/export/deepfly3d_v4_to_tracktail.py` |
| `scripts/check_status.py` | `scripts/dataset_tools/check_status.py` |
| `scripts/combine_roots.py` | `scripts/dataset_tools/combine_roots.py` |
| `scripts/compress_video_groups.py` | `scripts/dataset_tools/compress_video_groups.py` |
| `scripts/render_dataset.py` | `scripts/dataset_tools/render_dataset.py` |
| `scripts/diagnose_duplicate_tracks.py` | `scripts/diagnostics/diagnose_duplicate_tracks.py` |
| `scripts/audit_detector_events.py` | `scripts/diagnostics/audit_detector_events.py` |
| `scripts/eval_detector.py` | `scripts/diagnostics/eval_detector.py` |
| `scripts/eval_detector_trace.py` | `scripts/diagnostics/eval_detector_trace.py` |
| `scripts/scorer_error_correlation.py` | `scripts/diagnostics/scorer_error_correlation.py` |
| `scripts/scorer_shortcut_diagnostics.py` | `scripts/diagnostics/scorer_shortcut_diagnostics.py` |
| `scripts/render_scored_session.py` | `scripts/diagnostics/render_scored_session.py` |
| `scripts/merge_scorer_qc.py` | `scripts/diagnostics/merge_scorer_qc.py` |
| `scripts/lint_comments.py` | `scripts/tools/lint_comments.py` |
| `scripts/job_l4.sh` | `scripts/cluster/job_l4.sh` |
| `scripts/run_deeperfly_l4_session.sh` | `scripts/cluster/run_deeperfly_l4_session.sh` |
| `scripts/submit_deeperfly_l4.sh` | `scripts/cluster/submit_deeperfly_l4.sh` |
| `configs/deeperfly_ramdya_lsf_900.toml` | `scripts/cluster/deeperfly_ramdya_lsf_900.toml` |
