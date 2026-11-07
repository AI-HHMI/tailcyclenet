"""Post-hoc identity relabelling of a 2D prediction session (TrackGraph v7 + the v8 pixel-ID gate).

``graph``   the tracklet graph solver (Dawid-Skene witness calibration, path cover, stage-2 merge, pixel anchor).
``witness`` label-free witnesses: tracktail carry labels / edge table, pixel-ID kappa and self-calibrated C0.
``gate``    the frozen gate (v7 bar OR McNemar + null) and ``run_identity``, the one entry point.
"""
