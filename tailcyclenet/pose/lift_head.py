"""The single-view lift head (report 83 §9-§10): camera-frame 3D from ONE camera's predicted 2D.

A set-lifter over keypoint tokens. Each camera of a 3D window is lifted on its own: its token per
keypoint is a learned embedding of the GLOBAL registry id + a projection of the model's own
predicted 2D (DETACHED, undistorted and K^-1-normalised, centred and scaled per (camera, frame))
and a presence flag; one global token carries the 2D centroid ray and the log 2D scale. An
8-layer pre-norm transformer outputs the root log-depth (global token) and a per-keypoint depth
offset along its own ray, so the output is camera-frame XYZ = ray * depth.

Inputs are the 2D only (no decoder latent): report 83 §9 measured the 8 x 256 2D-only head at
Procrustes 6.3 mm vs 6.95 for 2 layers, at no 2D cost; an attached latent buys ~7 mm raw for
+0.55 px 2D, a detached one is worse. The constants (x5 centroid, +3 log scale, exp(. + log 800),
x30 offsets; the loss's /10 and x10) are the reference patch's (`scratch/singleview3d/patches/
sv3d_run.py`, SV_LIFT): mm-scaled priors that a learnable bias absorbs for other units.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from tracktail.model.cube import undistort_points


class LiftHead(nn.Module):
    """Set-lifter: normalised 2D keypoints of one (camera, frame) -> camera-frame 3D."""

    def __init__(self, n_keypoints: int, width: int = 256, layers: int = 8, heads: int = 8):
        """n_keypoints -- registry size (the embedding is keyed on the GLOBAL id, so `warm_start`
        maps its rows by name like the query encoder's); width/layers -- the transformer."""
        super().__init__()
        self.kpt_embed = nn.Embedding(n_keypoints, width)
        self.inp = nn.Linear(3, width)
        self.glob = nn.Linear(3, width)
        self.cls = nn.Parameter(torch.zeros(width))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, 4 * width, 0.1, batch_first=True,
                                       norm_first=True), layers, enable_nested_tensor=False)
        self.out_offset = nn.Linear(width, 1)
        self.out_root = nn.Linear(width, 1)

    def forward(self, n: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """n (N,K,2) normalised image coords (NaN = absent), ids (N,K) global ids -> (N,K,3)."""
        m = torch.isfinite(n).all(-1)
        n0 = torch.where(m[..., None], n, 0.)
        mf = m[..., None].float()
        cnt = mf.sum(1).clamp_min(1)
        c = (n0 * mf).sum(1) / cnt
        sc = ((((n0 - c[:, None]) ** 2).sum(-1) * m).sum(1) / cnt[:, 0]).sqrt().clamp_min(1e-4)
        x = ((n0 - c[:, None]) / sc[:, None, None]) * mf
        tok = self.kpt_embed(ids) + self.inp(torch.cat([x, mf], -1))
        g = self.cls + self.glob(torch.cat([c * 5, torch.log(sc)[:, None] + 3], -1))
        pad = torch.cat([torch.zeros_like(m[:, :1]), ~m], 1)
        h = self.enc(torch.cat([g[:, None], tok], 1), src_key_padding_mask=pad)
        z = (torch.exp(self.out_root(h[:, 0])[:, 0] + math.log(800.))[:, None]
             + self.out_offset(h[:, 1:])[..., 0] * 30.0)
        return torch.cat([n0 * z[..., None], z[..., None]], -1)


def static_cameras(camera_group) -> bool:
    """True when every camera has one (4,4) extrinsic (the head is not defined on a moving rig)."""
    return all(c['ext'].ndim == 2 for c in camera_group)


def lift_cameras(head: LiftHead, p2d: torch.Tensor, kpt_ids: torch.Tensor, camera_group):
    """Lift every camera of a window on its own -> (camera-frame, world), each (C,B,T,K,3).

    p2d (C,B,T,K,2) is the model's predicted 2D in crop pixels; it is DETACHED here (the head
    never trains the 2D). Undistortion uses the camera's crop offset at frame 0, as the reference
    did (a window's crop does not move). All cameras go through the head in one batch.
    """
    p2 = p2d.detach().float()
    C, B, T, K, _ = p2.shape
    ns = []
    for ci, cam in enumerate(camera_group):
        off = cam['offset'] if cam['offset'].ndim == 1 else cam['offset'][0]
        ns.append(undistort_points(dict(cam, offset=off), p2[ci].reshape(-1, 2))
                  .reshape(B * T, K, 2))
    ids = kpt_ids.to(p2.device).long()[:, None].expand(B, T, K).reshape(B * T, K)
    xc = head(torch.cat(ns, 0), ids.repeat(C, 1)).reshape(C, B, T, K, 3)
    E = torch.stack([cam['ext'][:3] for cam in camera_group]).to(xc.dtype)
    xw = torch.einsum('cji,cbtnj->cbtni', E[..., :3], xc - E[:, None, None, None, :, 3])
    return xc, xw


def lift_loss(lift_cam: torch.Tensor, coords_true: torch.Tensor, camera_group):
    """(total-before-weight, shape_mm, root) of the per-camera lift loss.

    lift_cam (C,B,T,K,3) camera frame; coords_true (B,T,K,3) world. Per (camera, item, frame)
    with >= 3 finite keypoints: shape = mean L1-norm of the centroid-aligned residual (world
    units, mm on every 3D root here), root = |log z(pred centroid) - log z(true centroid)|;
    total = shape / 10 + 10 * root, averaged over cameras by construction (each camera is its
    own sample). NaNs (detached) when nothing qualifies.
    """
    P = lift_cam
    E = torch.stack([c['ext'][:3] for c in camera_group]).to(P.dtype)
    G = (torch.einsum('cij,btnj->cbtni', E[..., :3], coords_true.to(P.dtype))
         + E[:, None, None, None, :, 3])
    m = torch.isfinite(G).all(-1) & torch.isfinite(P).all(-1)
    cnt = m.float().sum(-1, keepdim=True)
    ok = cnt[..., 0] >= 3
    nan = torch.tensor(float('nan'), device=P.device)
    if not ok.any():
        return nan, nan, nan
    mf = m[..., None].float()
    G0 = torch.where(m[..., None], G, 0.)
    P0 = torch.where(m[..., None], P, 0.)
    cg = (G0 * mf).sum(-2) / cnt.clamp_min(1)
    cp = (P0 * mf).sum(-2) / cnt.clamp_min(1)
    e = ((P0 - cp[..., None, :]) - (G0 - cg[..., None, :])).norm(dim=-1)
    shape = e[m & ok[..., None]].mean()
    root = (torch.log(cp[..., 2].clamp_min(1.))
            - torch.log(cg[..., 2].clamp_min(1.))).abs()[ok].mean()
    return shape / 10 + 10 * root, shape, root
