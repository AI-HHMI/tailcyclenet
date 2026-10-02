from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tailcyclenet.dataset import read_frames
from tailcyclenet.format import INST_LABELED, INST_PRESENT, Session, load_datasets

SAMPLES = 64
MAX_SIDE = 1024
MIN_VALID = 8


def _sample_frames(sess, count):
    """Choose evenly spaced frames from groups, including unlabeled frames."""
    groups = list(sess.groups.items())
    if not groups:
        return []
    indices = np.linspace(0, len(groups) - 1, count).round().astype(int)
    occurrences = {}
    frequencies = np.bincount(indices, minlength=len(groups))
    selected = []
    for index in indices:
        gid, group = groups[int(index)]
        occurrence = occurrences.get(int(index), 0)
        occurrences[int(index)] = occurrence + 1
        frame = int(round(occurrence * max(group.n_frames - 1, 0) /
                          max(frequencies[int(index)] - 1, 1)))
        selected.append((gid, frame))
    return selected


def _project(sess, lab, gid, frame, ci):
    """Return one animal-axis array of 2D points in camera pixels."""
    if sess.mode == '3d' and lab.points3d is not None:
        import torch
        from posetail.posetail.cube import project_points_torch

        cams = sess.__dict__.setdefault("_background_cameras", sess.rig.posetail())
        pts = torch.as_tensor(lab.points3d[:, frame], dtype=torch.float32)
        return project_points_torch([cams[ci]], pts)[0].cpu().numpy()
    if lab.points2d is not None:
        return np.asarray(lab.points2d[:, frame, :, ci])
    return np.empty((0, 0, 2), dtype=np.float32)


def _mask_for(sess, lab, gid, frame, ci, scale, height, width):
    """Build labeled foreground masks and retain their boxes for ghost checks."""
    mask = np.zeros((height, width), dtype=bool)
    boxes = []
    pts = _project(sess, lab, gid, frame, ci)
    if pts.ndim == 3:
        for animal in pts:
            finite = np.isfinite(animal).all(-1)
            if not finite.any():
                continue
            xy = animal[finite] * scale
            lo, hi = xy.min(0), xy.max(0)
            margin = 0.25 * max(float((hi - lo).max()), 1.0)
            box = np.array([lo[0] - margin, lo[1] - margin,
                            hi[0] + margin, hi[1] + margin])
            x0, y0, x1, y1 = np.floor(box).astype(int)
            x0, x1 = max(0, x0), min(width, x1 + 1)
            y0, y1 = max(0, y0), min(height, y1 + 1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
                boxes.append(box)
    if lab.boxes is not None and lab.instance is not None:
        for status, box in zip(lab.instance[:, frame, ci], lab.boxes[:, frame, ci]):
            if status not in (INST_LABELED, INST_PRESENT) or not np.isfinite(box).all():
                continue
            box = np.asarray(box, dtype=np.float32) * scale
            x0, y0, x1, y1 = np.floor(box).astype(int)
            x0, x1 = max(0, x0), min(width, x1 + 1)
            y0, y1 = max(0, y0), min(height, y1 + 1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
                boxes.append(box)
    return mask, boxes


def _robust_background(frames, label_masks):
    """Estimate a temporal median while iteratively excluding image foreground."""
    import cv2

    n, h, w, _ = frames.shape
    foreground = label_masks.copy()
    flat = frames.reshape(n, -1, 3)
    fgflat = foreground.reshape(n, -1)
    background = np.zeros((h * w, 3), dtype=np.uint8)
    threshold = np.full(h * w, 12.0, dtype=np.float32)
    for iteration in range(4):
        for start in range(0, h * w, 32768):
            end = min(start + 32768, h * w)
            blocked = fgflat[:, start:end]
            values = flat[:, start:end].astype(np.float32)
            values[blocked] = np.nan
            median = np.nanmedian(values, axis=0)
            background[start:end] = np.nan_to_num(median, nan=0).clip(0, 255).astype(np.uint8)
            if iteration < 3:
                deviation = np.abs(values - median[None])
                mad = np.nanmedian(deviation, axis=0).mean(-1)
                threshold[start:end] = np.maximum(12.0, 4.0 * np.nan_to_num(mad, nan=0.0))
        if iteration == 3:
            break
        bg = background.reshape(h, w, 3)
        limits = threshold.reshape(h, w)
        kernel = np.ones((5, 5), dtype=np.uint8)
        for i in range(n):
            delta = np.abs(frames[i].astype(np.float32) - bg).mean(-1)
            detected = delta > limits
            closed = cv2.morphologyEx(detected.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            foreground[i] |= cv2.dilate(closed, np.ones((3, 3), np.uint8)) > 0
            fgflat[i] = foreground[i].reshape(-1)
    counts = (~foreground).sum(0)
    return background.reshape(h, w, 3), counts >= MIN_VALID


def _session_job(task):
    """Estimate all camera backgrounds for one static training session."""
    session_path, out_root = task
    sess = Session.load(Path(session_path))
    if any(sess.rig.moving.values()):
        return {'session': sess.session_id, 'skipped': 'moving rig', 'views': {}}
    selected = _sample_frames(sess, SAMPLES)
    by_camera = {cam: [] for cam in sess.cam_names}
    for gid, frame in selected:
        group = sess.groups[gid]
        lab = sess.labels(gid)
        for ci, cam in enumerate(sess.cam_names):
            raw = read_frames(group, cam, [frame])[0]
            if raw is None:
                continue
            h0, w0 = raw.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h0, w0))
            if scale < 1.0:
                import cv2

                w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
            h, w = raw.shape[:2]
            mask, boxes = _mask_for(sess, lab, gid, frame, ci, scale, h, w)
            by_camera[cam].append((raw, mask, boxes))
    result = {}
    session_out = Path(out_root) / sess.session_id
    session_out.mkdir(parents=True, exist_ok=True)
    for cam, samples in by_camera.items():
        if not samples:
            continue
        frames = np.stack([x for x, _, _ in samples])
        masks = np.stack([m for _, m, _ in samples])
        background, valid = _robust_background(frames, masks)
        comparisons = np.abs(frames.astype(np.float32) - background[None]).mean(-1)
        ghost_boxes = 0
        for i, (_, _, boxes) in enumerate(samples):
            h, w = valid.shape
            for box in boxes:
                x0, y0, x1, y1 = np.floor(box).astype(int)
                x0, x1 = max(0, x0), min(w, x1 + 1)
                y0, y1 = max(0, y0), min(h, y1 + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                inside = comparisons[i, y0:y1, x0:x1].mean()
                outside_mask = np.ones(valid.shape, dtype=bool)
                outside_mask[y0:y1, x0:x1] = False
                outside = comparisons[i, outside_mask].mean() if outside_mask.any() else 0.0
                if inside <= min(outside, 12.0):
                    valid[y0:y1, x0:x1] = False
                    ghost_boxes += 1
        inside = masks.any(0)
        inside_delta = float(comparisons[:, inside].mean()) if inside.any() else 0.0
        outside_delta = float(comparisons[:, ~inside].mean()) if (~inside).any() else 0.0
        ghost_score = inside_delta / max(outside_delta, 1e-6)
        unknown_fraction = float((~valid).mean())
        usable = unknown_fraction < 0.25
        background[~valid] = 0
        Image.fromarray(background).save(session_out / f'{cam}.png')
        Image.fromarray((valid * 255).astype(np.uint8)).save(session_out / f'{cam}_valid.png')
        result[cam] = {'n_samples': len(samples), 'unknown_frac': unknown_fraction,
                       'ghost_score': ghost_score, 'ghost_boxes': ghost_boxes,
                       'inside_delta': inside_delta, 'outside_delta': outside_delta,
                       'usable': bool(usable), 'scale': min(1.0, MAX_SIDE / max(background.shape[:2])),
                       'size': [background.shape[1], background.shape[0]]}
    return {'session': sess.session_id, 'views': result}


def _contact_sheets(out_root, root_name, rows):
    """Render background thumbnails with unknown pixels highlighted red."""
    import cv2

    entries = []
    for row in rows:
        session = row['session']
        for camera, info in row['views'].items():
            image_path = Path(out_root) / root_name / session / f'{camera}.png'
            valid_path = Path(out_root) / root_name / session / f'{camera}_valid.png'
            image = np.asarray(Image.open(image_path).convert('RGB'))
            valid = np.asarray(Image.open(valid_path)) > 0
            thumb = cv2.resize(image, (160, 110), interpolation=cv2.INTER_AREA)
            vm = cv2.resize(valid.astype(np.uint8), (160, 110), interpolation=cv2.INTER_NEAREST) > 0
            thumb[~vm] = (0.65 * thumb[~vm] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
            entries.append((session, camera, info, thumb))
    per_sheet = 48
    for page, start in enumerate(range(0, len(entries), per_sheet), 1):
        subset = entries[start:start + per_sheet]
        sheet = Image.new('RGB', (6 * 180, 8 * 138), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (session, camera, info, thumb) in enumerate(subset):
            x, y = (i % 6) * 180 + 10, (i // 6) * 138
            sheet.paste(Image.fromarray(thumb), (x, y))
            draw.text((x, y + 112), f'{session[:12]} {camera} U{info["unknown_frac"]:.3f}', fill='black')
        sheet.save(Path(out_root) / f'{root_name}_contact_{page}.jpg', quality=90)


def main():
    """Run background estimation for each supplied dataset root."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--out', default='scratch/detector_transfer/backgrounds')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for data_root in args.data:
        datasets = load_datasets(Path(data_root), split='train')
        for ds in datasets:
            sessions = ds.sessions.get('train', [])
            tasks = [(str(s.path), str(out / ds.name)) for s in sessions]
            with mp.get_context('spawn').Pool(max(1, min(args.workers, 16))) as pool:
                rows = list(pool.imap_unordered(_session_job, tasks))
            rows.sort(key=lambda x: x['session'])
            views = {f'{row["session"]}/{cam}': info for row in rows
                     for cam, info in row['views'].items()}
            index = {'root': ds.name, 'split': 'train', 'samples_per_session': SAMPLES,
                     'views': views, 'skipped': [r for r in rows if 'skipped' in r],
                     'runtime_seconds': time.monotonic() - started}
            (out / ds.name).mkdir(parents=True, exist_ok=True)
            (out / ds.name / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
            _contact_sheets(out, ds.name, rows)
            usable = sum(v['usable'] for v in views.values())
            print(f'{ds.name}: {usable}/{len(views)} usable; {len(sessions)} sessions')
    print(f'elapsed_seconds={time.monotonic() - started:.1f}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
