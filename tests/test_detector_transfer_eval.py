import pytest

from tailcyclenet.detector.config import load_detector_config


def test_save_every_defaults_to_eval_every(tmp_path):
    config = tmp_path / 'detector.toml'
    config.write_text('[data]\npath = "dataset"\n[model]\nyolox = "hybrid"\n[training]\nout = "out"\neval_every = 13\n')
    cfg = load_detector_config(config)
    assert cfg['training']['save_every'] == 13


def test_save_every_rejects_nonpositive(tmp_path):
    config = tmp_path / 'detector.toml'
    config.write_text('[data]\npath = "dataset"\n[model]\nyolox = "hybrid"\n[training]\nout = "out"\nsave_every = 0\n')
    with pytest.raises(SystemExit, match='save_every'):
        load_detector_config(config)
