"""The reader cache's locking: a video read locks its own container, not the whole cache."""


def test_a_video_read_locks_per_container_not_globally():
    """`pixels._read_video` locks PER PATH: one global lock would kill `detect_group`'s multi-
    camera decode overlap (3.5x on 3dpop)."""
    from tailcyclenet.data import reader_cache as ds

    a, b = ds.read_lock_for('/x/cam0.mp4'), ds.read_lock_for('/x/cam1.mp4')
    assert a is not b
    assert ds.read_lock_for('/x/cam0.mp4') is a
    with a:                       # holding one container's lock must not block another's
        assert b.acquire(blocking=False)
        b.release()
