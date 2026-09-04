"""A standalone crop-level ReID net -- the pivot from the dense per-anchor embedding head.

Report 55: the dense head (`embed_dim`, `65074fe`) pools per-anchor features OVER the anchors a
detected box owns, so its embedding inherits whatever the detector's shared feature map already
carries there -- entangled with objectness/box-regression features, and, per the cross-camera
probe (`da5ccf5`/`f823ede`), camera-viewpoint-confounded (Cam1 improved, Cam2 got WORSE; forcing
cross-view positives did not fix it). Owner decision after three training runs: pivot to the
standard person/animal-ReID formulation instead -- a SEPARATE small net whose only input is one
animal's CROP PIXELS, no detector-head entanglement, no anchor-ownership question at all.

This module is deliberately PURE, mirroring `reid_loss.py`'s own contract: no training loop, no
optimizer, no wiring. `CropReidNet` -> raw (B, embed_dim) vectors; feed them straight to
`reid_loss.contrastive_loss(vectors, labels)`, unmodified -- that loss already only wants
`(vectors, labels)`, so nothing about it is anchor-specific and nothing here duplicates it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import letterbox


class CropReidNet(nn.Module):
    """A small from-scratch CNN over one animal crop -> a raw embedding vector.

    Four stride-2 conv stages (32/64/128/128 channels), global average pool, linear head. Small
    and trained from scratch (no pretrained backbone) on purpose: this net's only job is
    appearance identity on a small, low-diversity population (report 55: 3dpop is 18 individuals
    total, no cross-session identity training -- CLAUDE.md), not general-purpose vision.
    """

    def __init__(self, embed_dim=32):
        """`embed_dim` -- output vector length. No default in the caller: this project never
        ships an unstated identity-branch width (CLAUDE.md gotcha 11's own lesson, `gridresid_
        offset` had no default and raised; the same discipline applies to a new identity knob).
        """
        super().__init__()
        chans = (32, 64, 128, 128)
        layers = []
        c_in = 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1),
                      nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.stem = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, embed_dim)

    def forward(self, x):
        """`x` -- (B, 3, H, W) float, any consistent scale (`CropReidDataset` emits [0, 1]).

        Output (B, embed_dim) RAW vectors -- `contrastive_loss` does its own L2 normalisation,
        same convention the dense head's pooled vectors already use.
        """
        f = self.stem(x)
        f = f.mean(dim=(2, 3))
        return self.head(f)


class CropReidDataset(torch.utils.data.Dataset):
    """One item = one animal's crop, labelled for `contrastive_loss`. Wraps a `BoxDataset`.

    Reuses `BoxDataset._load_letterbox` -- THE existing one-decode-per-item entry point (also
    mosaic-lite's source) -- for the full frame and its per-animal letterboxed target boxes, so
    there is no second video-decode path and no second crop rule to drift from the shipped one.
    Each finite animal box is then re-cropped out of that already-decoded frame and RE-letterboxed
    to `crop_wh`, its own square input, independent of the detector's `input_wh`.

    Labelled by `(session_id, group, animal_row)`, remapped to a dense int id at construction.
    That key persists across every frame and camera for the SAME animal -- the format's own
    row-identity contract (`docs/annotation_format.md`) -- so two crops sharing a label really are
    the same physical animal. One canonical id table across the whole dataset is a bookkeeping
    convenience only: `contrastive_loss` only needs labels to agree WITHIN a batch, and CLAUDE.md
    is explicit that no cross-session identity training happens here (3dpop's 18 individuals are
    shared across every split, so a label is never asked to mean anything past its own dataset).
    """

    def __init__(self, boxes_ds, crop_wh=(96, 96)):
        """`boxes_ds` -- a constructed `BoxDataset`. `crop_wh` -- this net's own square input."""
        self.ds = boxes_ds
        self.crop_wh = crop_wh
        label_ids = {}
        entries = []
        for i, (sess, gid, f, ci) in enumerate(boxes_ds.index):
            boxes = boxes_ds.boxes_for(i)
            finite = torch.isfinite(boxes).all(-1)
            for row in torch.nonzero(finite).flatten().tolist():
                key = (sess.session_id, str(gid), row)
                label_ids.setdefault(key, len(label_ids))
                entries.append((i, row, label_ids[key]))
        self.entries = entries
        self.n_labels = len(label_ids)

    def __len__(self):
        """One entry per (item, animal-row) with a finite box -- NOT `len(self.ds)`."""
        return len(self.entries)

    def __getitem__(self, k):
        """(crop, label) -- crop is (3, H, W) float in [0, 1], H, W = `self.crop_wh` reversed.

        A degenerate re-crop (box clamps to nothing, which should not happen for a box
        `boxes_for` itself emitted, but the frame edge can still zero it out under rounding)
        returns a flat grey chip rather than raising -- the same "no signal here" convention
        `letterbox`'s own padding already uses, so a rare edge case costs one uninformative
        sample instead of stopping a training run.
        """
        i, row, label = self.entries[k]
        img, boxes, _ = self.ds._load_letterbox(i)
        box = boxes[row]
        x0, y0, x1, y1 = (int(v) for v in box.round().tolist())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            crop = np.full((self.crop_wh[1], self.crop_wh[0], 3), 114, np.uint8)
        else:
            crop, _, _ = letterbox(img[y0:y1, x0:x1], self.crop_wh)
        t = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        return t, label
