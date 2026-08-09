"""Does the model run, in every mode, and does the honest token actually fire?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch, tomllib
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.model import build_model

cfg = tomllib.load(open('configs/w9.toml', 'rb'))
R = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
lc = LoaderConfig(n_frames=4, image_size=256, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for name, want_mode in [('rat-city', '2d'), ('allen-mouse', '3d')]:
    ds = PoseDataset(R / name, 'val', lc)
    b = pose_collate([ds[0]])
    K = b.kpt_ids.shape[1]
    model = build_model(cfg['model'], n_keypoints=K).to(dev).eval()
    views = [v.to(dev) for v in b.views]
    cg = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()} for c in b.cgroup]
    with torch.no_grad():
        out = model(views, b.kpt_ids.to(dev), cg, mode=b.sample_info['mode'],
                    kpt_prior=b.kpt_prior.to(dev), prompt_time=b.prompt_t.to(dev))
    print(f'{name:12s} mode={b.sample_info["mode"]} cams={len(views)} '
          f'coords_pred={tuple(out["coords_pred"].shape)} finite='
          f'{torch.isfinite(out["coords_pred"]).float().mean():.3f}')
