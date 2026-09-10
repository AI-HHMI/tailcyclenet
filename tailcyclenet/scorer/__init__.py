"""Offline QC of tracked roots: a track-quality scorer trained on synthetic corruptions.

The model is `PoseScorer(PoseTrackerEncoder)` -- this repo's pose encoder plus an attention-pooling
head and a score/precision readout. It is NOT `posetail.ScorerEncoder`, and the reason is in
`model.py`: two classes both defining `encode_scene` cannot be combined, and the silent winner
would discard this repo's scene-speed path.

Relationship to the reference (`posetail-next`): the loss (`TripletScorerLoss`), the corruption
generators (`PointCorruptor` / `GENERATORS` / `apply_drop_mask`) and `AttentionPooling` are
imported verbatim from the installed `posetail`. The triplet ASSEMBLY is ours, because the
reference builds it against a loader whose contract this repo's loader does not share.
"""
from .model import PoseScorer, build_scorer

__all__ = ['PoseScorer', 'build_scorer']
