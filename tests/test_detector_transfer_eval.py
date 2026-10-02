import numpy as np
import pytest

from tailcyclenet.detector.data import BoxDataset, tt_photometric_transform
from tailcyclenet.detector.config import load_detector_config


def test_tt_gain_one_is_byte_identical():
    image = np.random.default_rng(3).integers(0, 256, (12, 18, 3), dtype=np.uint8)
    result = tt_photometric_transform(image, 1.0, 1.0)
    assert result is image
    assert np.array_equal(result, image)


def test_tt_transform_is_applied_only_when_requested(tiny_root):
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=1)
    transformed = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=1, tt_transform=lambda image: tt_photometric_transform(image, 0.5, 1.0))
    a = plain[0]['x']
    b = transformed[0]['x']
    assert not np.array_equal(a.numpy(), b.numpy())


def test_save_every_defaults_to_eval_every(tmp_path):
    config = tmp_path / 'detector.toml'
    config.write_text('[data]\npath = "dataset"\n[model]\nyolox = "tiny"\n[training]\nout = "out"\neval_every = 13\n')
    cfg = load_detector_config(config)
    assert cfg['training']['save_every'] == 13


def test_save_every_rejects_nonpositive(tmp_path):
    config = tmp_path / 'detector.toml'
    config.write_text('[data]\npath = "dataset"\n[model]\nyolox = "tiny"\n[training]\nout = "out"\nsave_every = 0\n')
    with pytest.raises(SystemExit, match='save_every'):
        load_detector_config(config)
