"""Suite-wide fixtures: the tiny synthetic roots, small model widths, cached provenance."""

import numpy as np
import pytest
import torch

from tailcyclenet import format as fmt
from tests.helpers import (
    _scene,
    _session_2d,
    _session_2d_tracked_dense,
    _session_3d,
    _session_centred_labels,
    _session_projected,
)

# Cap the intraop pool, or `-n` makes the suite slower: `nproc`-wide torch pool vs many xdist workers.
torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def _small_hybrid_detector(monkeypatch):
    """Build every test `hybrid` detector at a test width: the shipped one is ~54M params, and the
    detector training tests save several checkpoints each, which filled /tmp under `-n 32`.

    Only widths change (backbone base 8, one transformer block per stage, neck 32); the module
    graph and every code path are the shipped ones, and `load_detector` rebuilds the same width.
    """
    from tailcyclenet.detector import hybrid, yolox
    monkeypatch.setattr(yolox, 'NECK_WIDTH', 32)
    init = hybrid.HybridBackbone.__init__
    monkeypatch.setattr(init, '__defaults__', (8, (1, 1), 8, 4.0, True))


@pytest.fixture(autouse=True)
def _seeded_global_rngs():
    """Every test starts from the same torch / numpy-legacy / `random` state.

    Without it a test drawing `torch.rand` inputs sees whatever state the tests xdist happened to
    run before it left behind, so a property that holds for almost every draw fails at random.
    Entropy-seeded streams (`np.random.default_rng()`, train items) are untouched by design: a
    test over those must draw enough to fail with P < 1e-9."""
    import random

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(scope='session', autouse=True)
def _small_video_encoder():
    """Build every test `base` VJEPA encoder at width 128 with random weights.

    The shipped ViT-B is 86M params: each build read ~350 MB of pretrained weights from the hub
    cache, and each saved pose checkpoint was 350-700 MB. Depth stays 12 (the hierarchical taps
    are hard-coded for it) and head_dim stays 64 (the RoPE split). Session-scoped so that
    module-scoped model fixtures are built small too.
    """
    from functools import partial

    from torch import nn
    from tracktail.model import vjepa2

    shipped = vjepa2._make_vjepa2_1_encoder

    def make(model_name='vjepa2_1_vit_large_384', checkpoint_key='target_encoder', img_size=384,
             patch_size=16, tubelet_size=2, num_frames=64, pretrained=True, **kwargs):
        if model_name != 'vjepa2_1_vit_base_384':
            return shipped(model_name, checkpoint_key, img_size, patch_size, tubelet_size,
                           num_frames, pretrained, **kwargs)
        return vjepa2.VisionTransformer(
            patch_size=patch_size, img_size=(img_size, img_size), num_frames=num_frames,
            tubelet_size=tubelet_size, use_sdpa=True, use_silu=False, wide_silu=True,
            uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            embed_dim=128, depth=12, num_heads=2, mlp_ratio=2, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    vjepa2._make_vjepa2_1_encoder = make
    yield
    vjepa2._make_vjepa2_1_encoder = shipped


def pytest_configure(config):
    """Compute the git provenance ONCE, in the controller, before xdist spawns its workers (they
    inherit the environment). `git status` takes the index lock, so every checkpoint save running
    it concurrently serialised the suite: 32 concurrent calls on this NFS tree take ~13 s."""
    import json
    import os

    if 'TAILCYCLENET_TEST_PROVENANCE' not in os.environ:
        from tailcyclenet import checkpoints
        os.environ['TAILCYCLENET_TEST_PROVENANCE'] = json.dumps(checkpoints.provenance())


@pytest.fixture(scope='session', autouse=True)
def _cached_provenance():
    """Every `provenance()` call returns the controller's one answer (see `pytest_configure`)."""
    import json
    import os
    from importlib import import_module

    prov = json.loads(os.environ['TAILCYCLENET_TEST_PROVENANCE'])
    sites = [import_module(m) for m in ('tailcyclenet.checkpoints', 'tailcyclenet.infer.provenance',
                                        'tailcyclenet.infer.score', 'tailcyclenet.detector.train')]
    shipped = [s.provenance for s in sites]
    for s in sites:
        s.provenance = lambda: dict(prov)
    yield
    for s, f in zip(sites, shipped):
        s.provenance = f


@pytest.fixture(scope='session')
def tracked_no_assessment_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('tracked_dense')
    _session_2d_tracked_dense(root / 'catlike' / 'train' / 's')
    return root / 'catlike'


@pytest.fixture(scope='session')
def mixed_source_root(tmp_path_factory):
    """One root holding three of the four (mode, label_source) cells; `2d-tracked` is absent."""
    root = tmp_path_factory.mktemp('mixed') / 'ds'
    _session_2d(root / 'train' / 'a_2d_annot', label_source='annotated')
    _session_3d(root / 'train' / 'b_3d_annot', label_source='annotated')
    _session_3d(root / 'train' / 'c_3d_tracked', label_source='tracked')
    return root


@pytest.fixture(scope='session')
def tiny_root(tmp_path_factory):
    """A folder of two dataset roots -- the multi-dataset case."""
    root = tmp_path_factory.mktemp('tiny')
    _session_2d(root / 'ratlike' / 'train' / 'sess_a')
    _session_2d(root / 'ratlike' / 'val' / 'sess_b')
    _session_3d(root / 'mouselike' / 'train' / 'sess_c')
    _session_3d(root / 'mouselike' / 'train' / 'sess_moving', moving=True)
    return root


@pytest.fixture(scope='session')
def dataset_2d(tiny_root):
    return fmt.load_dataset(tiny_root / 'ratlike')


@pytest.fixture(scope='session')
def dataset_3d(tiny_root):
    return fmt.load_dataset(tiny_root / 'mouselike')


@pytest.fixture(scope='session')
def projected_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('projected')
    _session_projected(root / 'proj' / 'train' / 's')
    return root / 'proj'


@pytest.fixture(scope='session')
def centred_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('centred')
    _session_centred_labels(root / 'mid' / 'train' / 's')
    return root / 'mid'


@pytest.fixture(scope='session')
def dense_root(tmp_path_factory):
    """A fully-labelled 32-frame group -- long enough for a strided window to have room."""
    root = tmp_path_factory.mktemp('dense')
    _session_centred_labels(root / 'dense' / 'train' / 's', T=32, labelled=tuple(range(32)))
    return root / 'dense'


@pytest.fixture(scope='module')
def rat(tmp_path_factory):
    """2D multi-animal session, T=4 (one window)."""
    return _scene(tmp_path_factory, 'rat', 4)


@pytest.fixture(scope='module')
def mv(tmp_path_factory):
    """3D moving-rig session, T=4 (one window)."""
    return _scene(tmp_path_factory, 'mv', 4)
