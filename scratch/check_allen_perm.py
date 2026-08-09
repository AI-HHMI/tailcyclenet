"""Is the column-sort permutation right? Bone lengths decide it.

The transposition swaps X <-> X-base for 8 finger pairs. Those are the SHORTEST bones in the
skeleton (a fingertip to its base). If we apply the wrong ordering, exactly those edges get
their endpoints crossed and their lengths blow up, while the rest of the skeleton is untouched.
So: median bone length per edge, with and without the permutation.
"""
import sys, tomllib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt
from scripts.convert_v4 import column_sort_perm

spec = tomllib.load(open('configs/datasets/allen-mouse.toml', 'rb'))
names = list(spec['names']); K = len(names)
perm = column_sort_perm(names)
ix = {n: i for i, n in enumerate(names)}
edges = [(ix[a], ix[b]) for a, b in spec['skeleton']]

raw = np.load('/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4/'
              'allen-mouse/test/mouse1/2024-12-03T10_47_14_ix120467/pose3d.npz')['pose'][0]
stored = [str(s) for s in np.load(
    '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4/'
    'allen-mouse/test/mouse1/2024-12-03T10_47_14_ix120467/pose3d.npz')['keypoints']]
assert stored == names

moved = [i for i in range(K) if perm[i] != i]
print(f'permutation moves {len(moved)} of {K} keypoints:')
print('  ', [f'{names[i]}<-{names[perm[i]]}' for i in moved[:6]], '...')

def bones(pose):
    out = {}
    for (a, b), (na, nb) in zip(edges, spec['skeleton']):
        d = np.linalg.norm(pose[:, a] - pose[:, b], axis=-1)
        d = d[np.isfinite(d)]
        if len(d):
            out[f'{na}-{nb}'] = np.median(d)
    return out

fixed, unfixed = bones(raw[:, perm]), bones(raw)
short = sorted(fixed, key=lambda k: fixed[k])[:8]
print(f'\n{"edge":34s} {"repaired":>9s} {"raw":>9s}   (mm)')
for k in short:
    print(f'{k:34s} {fixed[k]:9.2f} {unfixed[k]:9.2f}')
tot_f = np.mean([v for v in fixed.values()]); tot_u = np.mean([v for v in unfixed.values()])
print(f'\nmean bone length: repaired {tot_f:.2f} mm, raw {tot_u:.2f} mm')
sd_f = np.std([fixed[k] for k in fixed]); sd_u = np.std([unfixed[k] for k in unfixed])
print(f'std  bone length: repaired {sd_f:.2f} mm, raw {sd_u:.2f} mm')

# and the converted dataset must agree with the repaired array
sess = fmt.Session.load(Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
                             'tailcycle-datasets/allen-mouse/test/mouse1'))
lab = sess.labels('2024-12-03T10_47_14_ix120467')
got, want = lab.points3d[0], raw[:, perm].astype(np.float32)
m = np.isfinite(want).all(-1)
print(f'\nconverted matches repaired source: {np.allclose(got[m], want[m], atol=1e-4)} '
      f'({m.sum()} finite points)')
print(f'NaNs preserved: {bool((~np.isfinite(got[~m])).all())}')
