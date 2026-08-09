import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tomllib, torch, numpy as np
from posetail.posetail.losses import TotalLoss
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.model import build_model
from tailcyclenet.checkpoints import warm_start, resolve_checkpoint
sys.path.insert(0, 'scripts'); from train import run_batch

cfg = tomllib.load(open('configs/w9.toml','rb'))
lc = LoaderConfig(**{k:v for k,v in cfg['data'].items() if k in LoaderConfig.__dataclass_fields__})
ds = PoseDataset(cfg['data']['path'], 'train', lc)
model = build_model(cfg['model'], n_keypoints=ds.registry.n_keypoints)
fresh = warm_start(model, resolve_checkpoint(Path(cfg['training']['checkpoint_path'])), verbose=False)
for n,p in model.named_parameters():
    if n.startswith('scene_encoder.encoder.'): p.requires_grad_(False)
model = model.to('cuda:0')
loss_fn = TotalLoss(**cfg['training']['losses'])

nan_loss = nan_grad = ok = 0
for i in range(8):
    b = pose_collate([ds[i % len(ds)]])
    loss, out = run_batch(model, loss_fn, b, 'cuda:0')
    si = b.sample_info
    tag = f"{si['mode']}{'/1cam' if si['single_view'] else ''}"
    if not torch.isfinite(loss):
        nan_loss += 1
        # which output is bad?
        bad = {k: float(torch.isfinite(v).float().mean()) for k,v in out.items()
               if torch.is_tensor(v) and v.is_floating_point()}
        print(f'{i} {tag} LOSS NaN  coords finite={torch.isfinite(b.coords).all(-1).float().mean():.3f} '
              f'outputs<1: {[k for k,v in bad.items() if v<1.0]}')
        continue
    model.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
    if not torch.isfinite(gn):
        nan_grad += 1
        bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f'{i} {tag} GRAD NaN loss={float(loss):.3f} first bad params: {bad[:5]} (of {len(bad)})')
    else:
        ok += 1
        print(f'{i} {tag} ok loss={float(loss):.3f} gn={float(gn):.1f}')
print(f'\nloss-nan {nan_loss}  grad-nan {nan_grad}  ok {ok}')
