"""The identity relabel writes animal_id only (and deletes the row-frames it cannot place)."""
import numpy as np
import polars as pl

from tailcyclenet.format import DICT_COLS, write_table
from tailcyclenet.identity.stage import label_names, recipe_for, relabel_tables


def _table(path, rows):
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    cols = {k: (np.asarray(v) if k in ('frame', 'x', 'y') else v) for k, v in cols.items()}
    write_table(path, cols, dict_cols=DICT_COLS)


def test_relabel_writes_only_animal_id(tmp_path):
    rows = []
    for g in ('g0', 'g1'):
        for t in range(4):
            for a in ('det00', 'det01'):
                for bp in ('nose', 'tail'):
                    rows.append(dict(group_id=g, frame=t, animal_id=a, bodypart=bp,
                                     x=float(t * 10 + (a == 'det01')), y=float(len(bp))))
    _table(tmp_path / 'points2d.pq', rows)
    before = pl.read_parquet(tmp_path / 'points2d.pq')
    lab = np.array([[0, 0, 1, 1], [1, 1, 0, -1]])          # a swap at frame 2; det01 dropped at frame 3
    counts = relabel_tables(tmp_path, 'g0', ['det00', 'det01'], lab)
    after = pl.read_parquet(tmp_path / 'points2d.pq')
    assert counts['points2d'] == (14, 2)
    drop = ((before['group_id'] == 'g0') & (before['frame'] == 3) & (before['animal_id'] == 'det01')).to_numpy()
    kept = before.filter(~pl.Series(drop))
    assert after.columns == before.columns and after.height == kept.height
    for c in after.columns:
        if c != 'animal_id':
            assert after[c].equals(kept[c]), c
    other = (kept['group_id'] == 'g1').to_numpy()
    assert after['animal_id'].to_numpy()[other].tolist() == kept['animal_id'].to_numpy()[other].tolist()
    g0 = after.filter(pl.col('group_id') == 'g0')
    got = {(f, a) for f, a in zip(g0['frame'].to_list(), g0['animal_id'].to_list())}
    assert got == {(0, 'id00'), (0, 'id01'), (1, 'id00'), (1, 'id01'), (2, 'id01'), (2, 'id00'), (3, 'id01')}
    swapped = g0.filter((pl.col('frame') == 2) & (pl.col('x') == 20.0))['animal_id'].unique().to_list()
    assert swapped == ['id01']


def test_label_names_and_recipes():
    assert label_names(np.array([[0, 12]]))(3) == 'id03'
    assert label_names(np.array([[0, 123]]))(3) == 'id003'
    assert recipe_for(['nose', 'left_ear', 'tail_base']) == 'keypoint'
    assert recipe_for(['centroid']) == 'box'
