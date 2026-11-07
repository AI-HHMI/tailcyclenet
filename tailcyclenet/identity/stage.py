"""The identity post-pass (`--identity`): pixel-ID crops -> per-video pixel-ID net -> witnesses -> the v8
gate -> the F2 + C0 relabel when the gate opens, base rows when it does not.

Runs after `SessionWriter.close()` on the written 2D prediction session, per group, reading the tt carry
records the loop wrote (`identity_tt.pkl`, `identity.carry`). The relabel edits the session's tables IN
PLACE, as `--identity-bridge` does: only `animal_id` changes, plus the deletion of the row-frames the
relabel cannot place (a row-frame with any non-finite keypoint: the stack solves on fully finite rows
only). Every decision and its inputs go to `identity.json` beside the tables and to `[provenance]`.

Crop recipe (report 76 v7 / PREREG7): `--identity-mask-head` -> mask crops; else keypoint-aligned crops
when the skeleton has head and tail-base keypoints (K >= 2), else detector-box crops (K = 1).
"""
from __future__ import annotations

import json
import pickle
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

TT_FILE = 'identity_tt.pkl'
REPORT_FILE = 'identity.json'
PIXID_FILE = 'identity_pixid_{gid}.npz'
LABELS_FILE = 'identity_labels_{gid}.npz'


def _source_notes(src, gid):
    """The source session group's `notes` (idtracker / UDMT roots carry their body length there)."""
    import pandas as pd
    f = Path(src) / 'groups.pq'
    if not f.exists():
        return None
    g = pd.read_parquet(f)
    if 'notes' not in g.columns:
        return None
    g = g[g.group_id.astype(str) == str(gid)]
    return None if g.empty or g.notes.isna().all() else str(g.notes.iloc[0])


def recipe_for(names, mask_head):
    """'mask' | 'keypoint' | 'box' (module docstring)."""
    from .crops import axis_parts
    if mask_head is not None:
        return 'mask'
    hd, tb = axis_parts([str(n) for n in names])
    return 'keypoint' if (len(names) >= 2 and hd and tb) else 'box'


def label_names(lab):
    """Identity label -> animal_id: `id00`.. for the pixel-ID classes and fresh ids alike."""
    width = max(2, len(str(int(lab.max())))) if lab.size and lab.max() >= 0 else 2
    return lambda v: f'id{int(v):0{width}d}'


def relabel_tables(path, gid, animal_ids, lab):
    """Rewrite `animal_id` of one group's rows from `lab` (S, T) (rows indexed like `animal_ids`).

    Row order and every other column survive; a row-frame whose label is -1 is deleted; rows at
    frames >= T (outside the identity range) are left alone. Returns per-table (relabelled, deleted)."""
    import polars as pl

    from ..format import DICT_COLS, write_table
    path = Path(path)
    row = {str(a): i for i, a in enumerate(animal_ids)}
    name = label_names(lab)
    S, T = lab.shape
    counts = {}
    for stem in ('points2d', 'points3d', 'instances'):
        f = path / f'{stem}.pq'
        if not f.exists():
            continue
        table = pl.read_parquet(f)
        if not {'animal_id', 'frame', 'group_id'}.issubset(table.columns):
            continue
        animal = np.asarray([str(v) for v in table['animal_id'].to_list()], dtype=object)
        frame = table['frame'].to_numpy().astype(np.int64)
        mine = np.asarray([str(v) == str(gid) for v in table['group_id'].to_list()])
        s = np.asarray([row.get(a, -1) for a in animal])
        sel = mine & (s >= 0) & (frame < T)
        new = animal.copy()
        lv = np.full(len(animal), -2)
        lv[sel] = lab[s[sel], frame[sel]]
        keep = ~(sel & (lv < 0))
        for i in np.flatnonzero(sel & (lv >= 0)):
            new[i] = name(lv[i])
        out = {}
        for c in table.columns:
            ser = table[c]
            if c == 'animal_id':
                out[c] = new[keep]
            elif ser.dtype in (pl.Float32, pl.Float64) or (ser.dtype.is_integer() and ser.null_count() == 0) \
                    or (ser.dtype == pl.Boolean and ser.null_count() == 0):
                out[c] = ser.to_numpy()[keep]
            else:
                out[c] = [v for v, k in zip(ser.to_list(), keep) if k]
        tmp = f.with_suffix('.pq.tmp')
        write_table(tmp, out, dict_cols=DICT_COLS)
        tmp.replace(f)
        counts[stem] = (int((sel & (lv >= 0)).sum()), int((~keep).sum()))
    return counts


def _write_provenance(path, report):
    """Merge the identity decision into `[provenance]` (scalars; per group when the session has several)."""
    import toml
    cfg_path = Path(path) / 'session.toml'
    with open(cfg_path, 'rb') as f:
        doc = tomllib.load(f)
    prov = doc.setdefault('provenance', {})
    prov['identity_report'] = REPORT_FILE
    groups = report['groups']
    for gid, g in groups.items():
        pre = 'identity_' if len(groups) == 1 else f'identity_{gid}_'
        prov[pre + 'on'] = bool(g['on'])
        prov[pre + 'arm'] = str(g['arm'] or '')
        prov[pre + 'crops'] = str(g['crops'])
        prov[pre + 'note'] = str(g['inputs'].get('note', ''))
        for k in ('kappa_uf', 'pix_acc_cal', 'base_acc_cal', 'C0', 'gate_ttS', 'n_ttSAME', 'bar', 'b', 'c',
                  'p_mcn', 'p_null', 'N', 'n_nodes', 'bl'):
            v = g['inputs'].get(k, g.get(k))
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                v = -1.0 if k not in ('C0',) else False
            prov[pre + k] = bool(v) if isinstance(v, (bool, np.bool_)) else (
                int(v) if isinstance(v, (int, np.integer)) else float(v))
    tmp = cfg_path.with_suffix('.toml.tmp')
    tmp.write_text(toml.dumps(doc))
    tmp.replace(cfg_path)


def _jsonable(x):
    """numpy scalars -> python, arrays dropped (the report holds scalars only)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def identity_group(out, gid, tt, *, stop=None, mask_head=None, detector=None, mask_side=1.6, device='cuda',
                   log=print):
    """One group: crops -> pixel-ID -> gate -> labels. Returns (record, labels or None, row animal_ids).

    The rows are the crops' rows: within [0, stop) a row with no entry is dropped (as the frozen stack
    cut its sessions), so `pred`, the pixel-ID net and the labels share one row axis."""
    from ..format.prediction import load_predictions
    from . import crops as C
    from .gate import run_identity
    from .pixid import save_pixid, train_pixid
    from .witness import body_length
    out = Path(out)
    meta = tomllib.load(open(out / 'session.toml', 'rb'))
    src = Path(meta['provenance']['source_session'])
    preds, _ = load_predictions(out, groups=[gid])
    d = next(iter(preds.values()))
    pred = d['pred'].astype(float)
    if stop:
        pred = pred[:, :int(stop)]
    names = list(meta['names'])
    recipe = recipe_for(names, mask_head)
    t0 = time.time()
    if recipe == 'mask':
        from .maskhead import load_maskhead
        c = C.mask_crops(out, detector, load_maskhead(mask_head, device), group=gid, device=device, stop=stop,
                         side=mask_side)
    elif recipe == 'keypoint':
        c = C.keypoint_crops(out, group=gid, stop=stop)
    else:
        c = C.box_crops(out, group=gid, stop=stop)
    t1 = time.time()
    ids = [str(x) for x in c['ids']]
    pos = {str(a): i for i, a in enumerate(d['animal_ids'])}
    pred = pred[[pos[i] for i in ids]]
    pix = train_pixid(c, device=device, log=lambda *a, **k: None)
    t2 = time.time()
    if pix.get('fail'):
        pix = None
    else:
        save_pixid(out / PIXID_FILE.format(gid=gid), pix)
    notes = _source_notes(src, gid)
    bl = body_length(pred, notes)
    bl_source = 'source notes' if (notes and ('body_length_px=' in notes or 'box_px=' in notes)) else 'keypoint extent'
    if not (bl > 0) and recipe != 'keypoint':
        bl, bl_source = float(C.box_body_length(out, gid, stop=stop)), 'median detector box side'
    recs = [r for r in tt['rec']['S'] if r['gid'] == gid]
    res = run_identity(pred, pix, recs, tt['events'].get(gid, []), bl)
    t3 = time.time()
    rec = dict(on=bool(res['on']), arm=res['arm'], crops=recipe, bl=float(bl), bl_source=bl_source,
               inputs=_jsonable(res['inputs']), pixid_failed=pix is None, T=int(pred.shape[1]),
               sec=dict(crops=t1 - t0, pixid=t2 - t1, gate=t3 - t2))
    log(f"identity {gid}: gate {'ON (' + str(res['arm']) + ')' if res['on'] else 'OFF'}; "
        + ', '.join(f'{k} {res["inputs"].get(k)}' for k in ('kappa_uf', 'C0', 'gate_ttS', 'n_ttSAME', 'b', 'c',
                                                          'p_mcn', 'p_null') if k in res['inputs'])
        + f"; crops {recipe} {t1 - t0:.0f}s, pixel-ID {t2 - t1:.0f}s, gate {t3 - t2:.0f}s", flush=True)
    return rec, (res['labels'] if res['on'] else None), ids


def identity_session(out, args, device):
    """`--identity` on a written session, in place (module docstring)."""
    out = Path(out)
    with open(out / TT_FILE, 'rb') as f:
        tt = pickle.load(f)
    det = None
    if getattr(args, 'identity_mask_head', None) is not None:
        from ..detector.checkpoint import load_detector
        det = load_detector(args.detector, device)
    import pandas as pd
    gids = [str(g) for g in pd.read_parquet(out / 'groups.pq').group_id.astype(str)]
    stop = int(getattr(args, 'end_frame', 0) or getattr(args, 'max_frames', 0) or 0) or None
    report = dict(rule='v8: C0 AND [v7 bar OR (p_mcn <= 0.01 AND p_null <= 0.05)]; ON -> F2 + C0 relabel',
                  tt=TT_FILE, groups={})
    for gid in gids:
        if not any(r['gid'] == gid for r in tt['rec']['S']) and gid not in tt['events']:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rec, lab, ids = identity_group(out, gid, tt, stop=stop, mask_head=args.identity_mask_head,
                                           detector=det, mask_side=getattr(args, 'identity_mask_side', 1.6),
                                           device=device)
        if lab is not None:
            np.savez_compressed(out / LABELS_FILE.format(gid=gid), labels=lab, ids=np.array(ids))
            rec['relabel'] = relabel_tables(out, gid, ids, lab)
        report['groups'][gid] = rec
    (out / REPORT_FILE).write_text(json.dumps(report, indent=1))
    _write_provenance(out, report)
    return report
