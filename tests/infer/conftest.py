"""Fixtures shared by the inference tests."""
import pytest


@pytest.fixture(scope='module', params=['rat', 'mv'])
def scene(request):
    """Both shapes, for the few invariants whose 2D and 3D code paths differ."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def cli():
    """The inference CLI module (`scripts/infer.py` just calls its `main`)."""
    from tailcyclenet.infer import cli as mod
    return mod
