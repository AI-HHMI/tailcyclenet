"""FROZEN, byte-for-byte snapshot of `PoseDataset._item` as of commit `918471c` (the commit this repo was at when the seam was landed).

NOT production code and never imported by the package. It exists so
`test_dataset_select_realise.py` can compare the `_select`/`_realise`/`_targets` split against
the REAL pre-refactor implementation instead of against itself -- a test that re-ran `_item`
would be vacuous, because `_item` only delegates to the three parts.

Regenerate (only if the pose loader is deliberately changed and the new baseline re-approved):

    python - <<'EOF'
    import subprocess
    src = subprocess.run(['git', 'show', 'HEAD:tailcyclenet/dataset.py'],
                         capture_output=True, text=True, check=True).stdout.split('\n')
    start = next(i for i, l in enumerate(src) if l.startswith('    def _item('))
    end = next(i for i, l in enumerate(src) if i > start and l.startswith('    def _augment('))
    print('\n'.join(src[start:end]).rstrip())
    EOF
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
from posetail.datasets.posetail_dataset import rotate_camera_image_plane_3d
from posetail.posetail.cube import (get_camera_scale, is_point_visible, project_points_torch)

from tailcyclenet import box_prompt as bpmod
from tailcyclenet import crop as cropmod
from tailcyclenet.dataset import (_apply_affine, _crop_inflate, _mask_outside, _resize_camera,
                                 _rotate_2d, _rotate_camera_group_with_neighbours, _vis2d_target,
                                 prior_out_of_bounds, read_frames)
from tailcyclenet.format import VISIBLE


class LegacyItemMixin:
    """`_item` exactly as it was before the seam. Mix in FIRST to shadow the new one."""

    def _item(self, idx, rng, shape=None):
        """Build one window's tensors, or None when the item cannot be built.

        Inputs: idx -- the index to pick.
                rng -- the item's RNG stream.
                shape -- a pre-drawn cost-determining shape dict (see `_shape`).
        Outputs: the 13- or 14-field item tuple `__getitem__` hands to the collate, or None.

        Visibility: `vis` (3D noisy-OR) is unused at R == 2; `vis_2d` is the per-camera
        THREE-STATE target (NaN = "not assessed", masked out of the BCE; `projected` joins
        UNLABELED), withheld when `sess.has_visibility_assessment` is False or every
        assessed row is projected/unlabeled; both-or-neither.

        Geometry: camera and crop-inflate draws are one per item (camera draw sorted). Points
        outside the source frame or the FINAL crop are flipped out of `vis_2d`; a rotation that
        loses the animal to the inscribed crop is REVERTED, not retried.

        Pixels: appearance augmentation runs on the final ~256 px crops; views are UINT8 (4x
        fewer bytes to collate/queue/pin; the model divides on device).

        The query prior: `kpt_prior` is the pose at the prompt frame (GT at training, the
        previous window's own prediction at deployment); `prompt_t` is the first labelled frame;
        `prompt_dropout` is PER ITEM, not per keypoint. Corruptions, in order: exposure bias,
        noise/offset in PIXELS, stale priors, `prompt_swap_animal`, `prompt_swap_kpt_pairs`
        (finite entries independently replaced by another keypoint's ORIGINAL position), and a
        whole-body offset -- ONE vector per item. `pose_only_prob` (0 = byte-identical) hoists
        the box-dropout coin EARLY so a box-dropped item can also skip these corruptions,
        REPLACING not adding to `box_prompt_dropout`'s fraction; the box-prompt block reuses it.

        The stride is read back off `frames` (MEDIAN: a group-edge window repeats its
        last frame); the final 3D noisy-OR is over the FINAL `vis_2d`.
        """
        shape = shape or self._shape(rng)
        item = self._pick(idx, rng)
        sess, group = item.session, item.session.groups[item.gid]
        lab = sess.labels(item.gid)
        frames = self._frames(item, lab, group, rng)
        if frames is None:
            return None
        a = item.animal
        K = sess.n_keypoints

        attempt_swap_animal = self.train and self.cfg.prompt_swap_animal > 0 and lab.n_animals >= 2
        neighbour_row = None
        if attempt_swap_animal:
            other_rows = [i for i in range(lab.n_animals) if i != a]
            neighbour_row = other_rows[int(rng.integers(len(other_rows)))]

        cgroup = sess.cgroup(item.gid, frames)
        inflate = _crop_inflate(self.cfg, rng, self.train)

        true_2d = sess.mode == '2d'
        single_view = (not true_2d and self.train
                       and self.cfg.prob_2d_only > 0
                       and shape['single_view_draw'] < self.cfg.prob_2d_only)

        if true_2d:
            cam_ix = [0]
        elif single_view:
            cam_ix = [int(rng.integers(len(cgroup)))]
        else:
            n = shape['n_cams']
            cam_ix = (sorted(rng.choice(len(cgroup), n, replace=False)) if 0 < n < len(cgroup)
                      else list(range(len(cgroup))))
        cgroup = [cgroup[i] for i in cam_ix]
        cam_names = [sess.cam_names[i] for i in cam_ix]
        crop_pts = self._crop_pts(lab, a, frames, cam_ix)

        if true_2d:
            coords = torch.as_tensor(lab.points2d[a][frames][:, :, 0], dtype=torch.float32)
            vis = vis_2d = None
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = _vis2d_target(v2)
                if not torch.isfinite(vis_2d).any():
                    vis_2d = None
        else:
            coords = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = _vis2d_target(v2)
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
                if not torch.isfinite(vis_2d).any():
                    vis = vis_2d = None
            else:
                vis = vis_2d = None

        if torch.isfinite(coords).all(-1).sum() < 2:
            return None

        rot_p = (self.cfg.aug_prob if self.cfg.aug_rotation_prob is None
                 else self.cfg.aug_rotation_prob)
        rot_deg = self.cfg.aug_rotation_deg
        rotation_info = [None] * len(cgroup)
        neighbour_full = None
        if true_2d:
            cam = cgroup[0]
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            cp = None if crop_pts is None else crop_pts[:, 0]
            if self.train and rng.random() < rot_p:
                cam, coords, rot = _rotate_2d(cam, coords,
                                              float(rng.uniform(-rot_deg, rot_deg)))
                rotation_info = [rot]
                cp = _apply_affine(cp, rot)
            jit = self._jitter(rng)
            cam, box, coords = cropmod.crop_to_points_2d(cam, coords, self.cfg.min_crop_dim,
                                                         jit, crop_pts=cp,
                                                         inflate=inflate)
            if cam is None:
                return None
            cam, scale = _resize_camera(cam, self.cfg.image_size)
            coords = coords * scale
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            if attempt_swap_animal:
                raw = torch.as_tensor(lab.points2d[neighbour_row][frames][..., 0, :],
                                      dtype=torch.float32)
                raw = _apply_affine(raw, rotation_info[0])
                neighbour_full = (raw - box[:2].to(raw.dtype)) * scale
            cgroup, boxes = [cam], [box]
            p2d = p2d_all = coords[None]
            R = 2
        else:
            if self.train:
                rotated = []
                for cam in cgroup:
                    if rng.random() < rot_p:
                        cam_r, rot = rotate_camera_image_plane_3d(
                            cam, float(rng.uniform(-rot_deg, rot_deg)))
                        if int(is_point_visible(cam_r, coords).sum()) < 2 <= int(
                                is_point_visible(cam, coords).sum()):
                            cam_r, rot = cam, None
                        rotated.append((cam_r, rot))
                    else:
                        rotated.append((cam, None))
                cgroup = [c for c, _ in rotated]
                rotation_info = [r for _, r in rotated]
                if vis_2d is not None:
                    for cnum, cam in enumerate(cgroup):
                        if rotation_info[cnum] is not None:
                            vis_2d[:, :, cnum][~is_point_visible(cam, coords)] = 0
            others_raw = None
            if attempt_swap_animal:
                others_raw = torch.as_tensor(lab.points3d[neighbour_row][frames],
                                             dtype=torch.float32)
            if self.train:
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            cp3 = None if crop_pts is None else [
                _apply_affine(crop_pts[:, i], rotation_info[i]) for i in range(len(cgroup))]
            jit = self._jitter(rng)
            cgroup, boxes = cropmod.crop_to_points_3d(cgroup, coords, self.cfg.min_crop_dim,
                                                      jit, crop_pts=cp3,
                                                      inflate=inflate)
            if cgroup is None:
                return None
            cgroup = [_resize_camera(c, self.cfg.image_size)[0] for c in cgroup]
            if self.train:
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            if others_raw is not None:
                neighbour_full = others_raw
            p2d_all = (project_points_torch(cgroup, coords)
                       if single_view or self._aug is not None else None)
            p2d = p2d_all if single_view else None
            R = 3

        gray = self._aug is not None and rng.random() < self.cfg.grayscale_prob
        use_pool = group.source(cam_names[0])[0] != 'video'
        with (ThreadPoolExecutor(max_workers=16) if use_pool else nullcontext()) as pool:
            views = []
            for cnum, cam_name in enumerate(cam_names):
                imgs = read_frames(group, cam_name, frames, crop_coords=boxes[cnum],
                                   target_size=cgroup[cnum]['size'].tolist(),
                                   rotation=rotation_info[cnum], pool=pool)
                if any(im is None for im in imgs):
                    return None
                if self._aug is not None:
                    imgs = self._augment(imgs, cnum, cgroup[cnum]['size'], p2d_all, vis_2d,
                                         gray, rng)
                views.append(torch.from_numpy(np.asarray(imgs)))

        box_dropped_early = None
        pose_only_active = False
        if self.train and self.cfg.pose_only_prob > 0 and self.cfg.box_prompt != 'none' \
                and self.cfg.box_prompt_dropout > 0:
            box_dropped_early = rng.random() < self.cfg.box_prompt_dropout
            if box_dropped_early:
                pose_only_active = rng.random() < self.cfg.pose_only_prob

        finite = torch.isfinite(coords).all(-1)
        prompt_t = torch.where(finite.any(0), finite.float().argmax(0), torch.zeros(K).long())
        prompt_t = prompt_t.to(torch.int32)
        kpt_prior = coords[prompt_t, torch.arange(K)].clone()
        kpt_prior[~finite.any(0)] = float('nan')
        dropped_fully = False
        if self.train and not pose_only_active and rng.random() < self.cfg.prompt_dropout:
            kpt_prior[:] = float('nan')
            dropped_fully = True
        px = 1.0
        if R == 3 and (self.cfg.prompt_noise_px > 0 or self.cfg.prompt_offset_px > 0) \
                and bool(torch.isfinite(kpt_prior).any()):
            pts = kpt_prior[torch.isfinite(kpt_prior).all(-1)][None]
            px = float(torch.nanmedian(get_camera_scale(cgroup, pts)))
            if not np.isfinite(px):
                px = 1.0
        if self.train and self.cfg.prompt_stale_frames > 0 \
                and bool(torch.isfinite(kpt_prior).any()):
            d = int(rng.integers(0, int(self.cfg.prompt_stale_frames) + 1))
            if d:
                t_alt = torch.clamp(prompt_t.long() + d, max=coords.shape[0] - 1)
                alt = coords[t_alt, torch.arange(K)]
                swap = torch.isfinite(alt).all(-1) & torch.isfinite(kpt_prior).all(-1)
                kpt_prior = torch.where(swap[:, None], alt, kpt_prior)
        if neighbour_full is not None:
            mode_str = '2d' if R == 2 else '3d'
            neighbour_prior = neighbour_full[prompt_t, torch.arange(K)].clone()
            neighbour_prior[prior_out_of_bounds(neighbour_prior, mode_str, cgroup)] = float('nan')
            jump = (torch.as_tensor(rng.random(K)) < self.cfg.prompt_swap_animal) \
                & torch.isfinite(neighbour_prior).all(-1)
            if not dropped_fully:
                kpt_prior = torch.where(jump[:, None], neighbour_prior, kpt_prior)
        if self.train and self.cfg.prompt_swap_kpt_pairs > 0:
            finite_idx = torch.isfinite(kpt_prior).all(-1).nonzero(as_tuple=True)[0]
            m = len(finite_idx)
            if m >= 2:
                sel = torch.as_tensor(rng.random(m)) < self.cfg.prompt_swap_kpt_pairs
                if bool(sel.any()):
                    local = torch.arange(m)[sel]
                    offset = torch.from_numpy(rng.integers(1, m, size=int(sel.sum())))
                    src_idx = finite_idx[(local + offset) % m]
                    original = kpt_prior
                    kpt_prior = kpt_prior.clone()
                    kpt_prior[finite_idx[sel]] = original[src_idx]
        if self.train and not pose_only_active and self.cfg.prompt_offset_px > 0 \
                and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_offset_px) * px, (1, R)),
                dtype=kpt_prior.dtype)
        if self.train and not pose_only_active and self.cfg.prompt_noise_px > 0 \
                and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_noise_px) * px, kpt_prior.shape),
                dtype=kpt_prior.dtype)

        kpt_ids = self._kpt_ids[sess.path]
        query_times = torch.zeros(K, dtype=torch.int32)
        query_occlusion = torch.full((K, len(cgroup)), -1, dtype=torch.int64)
        stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1

        if vis is not None and vis_2d is not None:
            vis = (vis_2d == 1).any(-1)

        row = {'dataset': self.datasets[item.ds].name, 'session': sess.session_id,
               'group': item.gid, 'animal': lab.animal_ids[a], 'mode': '2d' if R == 2 else '3d',
               'single_view': single_view, 'start': int(frames[0]), 'cameras': cam_names,
               'stride': stride}

        out = [views, coords, vis, torch.as_tensor(frames), cgroup, row, query_times,
               vis_2d, p2d, query_occlusion, kpt_ids, kpt_prior, prompt_t]
        if self.cfg.box_prompt != 'none':
            from tailcyclenet import box_prompt as bpmod   # namespace repaired: `.` would mean tests/
            box = bpmod.compute_box_prompt(coords, cgroup, '2d' if R == 2 else '3d')
            box = bpmod.apply_frames_mode(box, self.cfg.box_prompt_frames)
            if self.train:
                dropped = (box_dropped_early if box_dropped_early is not None
                          else (self.cfg.box_prompt_dropout > 0
                                and rng.random() < self.cfg.box_prompt_dropout))
                if dropped:
                    box = torch.full_like(box, float('nan'))
                elif self.cfg.box_prompt_jitter > 0 or self.cfg.box_prompt_scale_jitter > 0:
                    box = bpmod.apply_jitter(box, rng, self.cfg.box_prompt_jitter,
                                             self.cfg.box_prompt_scale_jitter)
            out.append(box)
        return tuple(out)
