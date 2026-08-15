"""tailcyclenet -- posetail finetuned into an animal pose estimator.

`patches.apply_all()` runs HERE, at package import, and not at the call sites that need it.
`posetail`'s own modules bind `project_cam` and `project_points_torch` by VALUE at their import
time, so a patch applied later than the first `import posetail.posetail.losses` would reach some
call sites and not others -- which is the worst of the three outcomes. Importing anything from
this package applies it; see `patches` for what each one is and what to send upstream.
"""
from . import patches as _patches

_patches.apply_all()
