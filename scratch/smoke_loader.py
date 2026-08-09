"""Drive the real converted datasets through the loader."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import Registry, load_datasets

R = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
cfg = LoaderConfig(n_frames=24, image_size=256)

for name in ['rat-city', 'allen-mouse', '3dpop', 'branson-fly']:
    t0 = time.time()
    ds = PoseDataset(R / name, 'train', cfg)
    t1 = time.time()
    modes, shapes = {}, set()
    for i in range(6):
        b = pose_collate([ds[i % len(ds)]])
        si = b.sample_info
        modes[si['mode'] + ('/1cam' if si['single_view'] else '')] = modes.get(
            si['mode'] + ('/1cam' if si['single_view'] else ''), 0) + 1
        shapes.add((len(b.views), tuple(b.views[0].shape), tuple(b.coords.shape),
                    None if b.vis is None else tuple(b.vis.shape)))
    print(f'{name:12s} index={len(ds):5d}  preload={t1-t0:5.1f}s  '
          f'item={(time.time()-t1)/6:5.2f}s  modes={modes}')
    for s in sorted(shapes, key=str):
        print(f'   cams={s[0]} views={s[1]} coords={s[2]} vis={s[3]}')

print('\n--- multi-dataset registry ---')
dss = load_datasets(R)
reg = Registry.build(dss)
print(f'{len(dss)} datasets, {reg.n_keypoints} keypoints total')
for n, ids in reg.datasets:
    print(f'  {n:12s} ids {ids[0]}..{ids[-1]}  e.g. {reg.names[ids[0]]!r}')
