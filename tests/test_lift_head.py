"""The single-view lift head (report 83 §9) and the warm-start registry spelling fix."""
from tailcyclenet.format import Registry


def test_spelled_as_maps_single_root_onto_prefixed_base():
    base = Registry(names=('a-nose', 'a-ear', 'b-nose'), datasets=(('a', (0, 1)), ('b', (2,))))
    mine = Registry(names=('nose', 'ear', 'tail'), datasets=(('a', (0, 1)), ('c', (2,))))
    assert mine.spelled_as(base) == ('a-nose', 'a-ear', 'tail')


def test_spelled_as_strips_prefix_for_unprefixed_base():
    base = Registry(names=('nose', 'ear'), datasets=(('a', (0, 1)),))
    mine = Registry(names=('a-nose', 'a-ear', 'b-x'), datasets=(('a', (0, 1)), ('b', (2,))))
    assert mine.spelled_as(base) == ('nose', 'ear', 'b-x')
    assert base.spelled_as(base) == base.names
