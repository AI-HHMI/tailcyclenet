import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tomllib, torch
from posetail.posetail.losses import TotalLoss
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.model import build_model
sys.path.insert(0, 'scripts'); from train import run_batch

cfg = tomllib.load(open('configs/w9.toml','rb'))
lc = LoaderConfig(**{k:v for k,v in cfg['data'].items() if k in LoaderConfig.__dataclass_fields__})
ds = PoseDataset(cfg['data']['path'], 'train', lc)
model = build_model(cfg['model'], n_keypoints=ds.registry.n_keypoints).to('cuda:0')
loss_fn = TotalLoss(**cfg['training']['losses'])
b = pose_collate([ds[0]])

# 1. are the forward outputs finite in train mode?
loss, out = run_batch(model, loss_fn, b, 'cuda:0')
print('loss', float(loss))
for k, v in out.items():
    if torch.is_tensor(v) and v.is_floating_point():
        f = torch.isfinite(v).float().mean()
        if f < 1.0: print(f'  output {k}: finite {f:.4f}')
if isinstance(out.get('grid'), dict):
    for k, v in out['grid'].items():
        if torch.is_tensor(v) and v.is_floating_point():
            f = torch.isfinite(v).float().mean()
            if f < 1.0: print(f'  grid.{k}: finite {f:.4f}')
print('coords finite:', float(torch.isfinite(b.coords).all(-1).float().mean()))

# 2. which op produces the NaN in backward
with torch.autograd.set_detect_anomaly(True):
    loss, out = run_batch(model, loss_fn, b, 'cuda:0')
    model.zero_grad()
    try:
        loss.backward()
        print('backward ok')
    except RuntimeError as e:
        print('ANOMALY:', str(e).split('\n')[0])
        for line in str(e).split('\n'):
            if 'Traceback' in line or '.py", line' in line or 'in forward' in line:
                print('   ', line.strip())
