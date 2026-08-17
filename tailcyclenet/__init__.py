"""tailcyclenet -- posetail finetuned into an animal pose estimator.

No monkeypatching: posetail >= 0.3.5 ships every behaviour this repo once had to patch in
(per-frame camera offsets, `crop_box_for_points`, `scene_features=`/`input_size=` on
`TrackerEncoder.forward`). See dev/reports/32_posetail_035_upgrade.md.
"""
