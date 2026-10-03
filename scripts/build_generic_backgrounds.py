"""Build `generic_backgrounds_coco_v1`: animal-free COCO images for detector background aug.

One-shot. Reads the COCO val2017 download (`--src`) and writes a curated pack (`--out`, default
$TAILCYCLENET_CACHE_DIR/generic_backgrounds_coco_v1): `images/*.jpg` (max side 512, JPEG q85),
`manifest.json` (source, id, licence, attribution URL per image) and one `.tar` archive.

Selection, deterministic:
- COCO val2017 images with NO annotation in the `animal` or `person` supercategories and no
  `teddy bear`, and only under licences that allow redistributing a resized copy: CC BY 2.0,
  CC BY-SA 2.0, "No known copyright restrictions" and US Government Work. NC and ND licences are
  dropped. Each kept image keeps its Flickr URL for attribution; BY-SA images stay BY-SA.
No DTD (owner, 2026-10-03): its images are research-use only and cannot ship. Report 73's runs
used the earlier `generic_backgrounds_v1` (this COCO set + 184 DTD textures).
COCO annotations are incomplete, so an unannotated person or animal can still appear; the pack is
for background/negative augmentation where a rare miss costs little.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 'generic_backgrounds_coco_v1'
MAX_SIDE = 512
QUALITY = 85
# COCO licence ids (instances_val2017.json `licenses`): 4 BY, 5 BY-SA, 7 no known copyright, 8 US Gov.
COCO_KEEP_LICENSES = {4, 5, 7, 8}
COCO_DROP_SUPERCATEGORIES = {'animal', 'person'}
COCO_DROP_NAMES = {'teddy bear'}


def _save(src: Path, dst: Path) -> tuple[int, int]:
    """Resize to MAX_SIDE (never upscale), save JPEG, return (w, h)."""
    im = Image.open(src).convert('RGB')
    s = min(1.0, MAX_SIDE / max(im.size))
    if s < 1.0:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    im.save(dst, quality=QUALITY)
    return im.size


def _coco(src: Path, n: int, rng: np.random.Generator):
    """Animal- and person-free COCO val2017 images under redistributable licences."""
    ann = json.loads((src / 'annotations' / 'instances_val2017.json').read_text())
    lic = {x['id']: x for x in ann['licenses']}
    drop = {c['id'] for c in ann['categories']
            if c['supercategory'] in COCO_DROP_SUPERCATEGORIES or c['name'] in COCO_DROP_NAMES}
    bad = {a['image_id'] for a in ann['annotations'] if a['category_id'] in drop}
    keep = sorted((im for im in ann['images']
                   if im['id'] not in bad and im['license'] in COCO_KEEP_LICENSES),
                  key=lambda im: im['id'])
    pick = keep if n <= 0 or n >= len(keep) else [keep[i] for i in sorted(
        rng.choice(len(keep), n, replace=False))]
    for im in pick:
        yield (src / 'val2017' / im['file_name'],
               {'source': 'coco_val2017', 'source_id': im['file_name'],
                'license': lic[im['license']]['name'], 'license_url': lic[im['license']]['url'],
                'attribution_url': im.get('flickr_url') or im.get('coco_url')})


def main():
    """Build the pack, its manifest and its archive."""
    cache = Path(os.environ.get('TAILCYCLENET_CACHE_DIR', Path.home() / '.cache' / 'tailcyclenet'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=cache / 'images')
    ap.add_argument('--out', type=Path, default=cache / VERSION)
    ap.add_argument('--coco', type=int, default=0, help='COCO images to keep; 0 = every eligible')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    img_dir = args.out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, meta in _coco(args.src, args.coco, rng):
        name = f'{len(rows):04d}_{meta["source"]}.jpg'
        w, h = _save(path, img_dir / name)
        meta.update(file=f'images/{name}', width=w, height=h,
                    sha256=hashlib.sha256((img_dir / name).read_bytes()).hexdigest())
        rows.append(meta)
    manifest = {'version': VERSION, 'max_side': MAX_SIDE, 'jpeg_quality': QUALITY,
                'seed': args.seed, 'n_images': len(rows),
                'counts': {'coco_val2017': len(rows)},
                'note': 'Training-only background/negative images. COCO filtered on its own '
                        '(incomplete) annotations: no animal, person or teddy bear; redistributable '
                        'licences only. Per-image licence and attribution below.',
                'images': rows}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    archive = args.out.parent / f'{VERSION}.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(args.out, arcname=VERSION)
    print(f'{len(rows)} images ({manifest["counts"]}) -> {args.out}; '
          f'archive {archive} {archive.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
