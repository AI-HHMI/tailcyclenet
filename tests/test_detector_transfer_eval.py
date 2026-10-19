import pytest

from tailcyclenet.detector.config import load_detector_config


def test_save_every_defaults_to_eval_every_and_must_be_positive(tmp_path):
    """An absent `save_every` follows `eval_every`; a non-positive one is refused by name."""
    config = tmp_path / 'detector.toml'
    head = '[data]\npath = "dataset"\n[model]\nyolox = "hybrid"\n[training]\nout = "out"\n'
    config.write_text(head + 'eval_every = 13\n')
    assert load_detector_config(config)['training']['save_every'] == 13
    config.write_text(head + 'save_every = 0\n')
    with pytest.raises(SystemExit, match='save_every'):
        load_detector_config(config)
