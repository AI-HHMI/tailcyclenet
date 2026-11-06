"""The v7 2D MoE through THE window loop: routing reaches the written rows, and a model without a
set decoder (or E1-only routing) is the plain path byte for byte."""
import numpy as np
import torch

from tailcyclenet.infer.window import run_group
from tailcyclenet.pose.model import build_model
from tests.helpers import SMALL, _infer_cfg, _scene

BOX = {**SMALL, 'box_prompt': 'film'}


def _boxes_for(boxes):
    def f(store, lo, hi):
        return (boxes[:, lo:hi], None)
    return f


def _assert_same(a, b):
    assert set(a) == set(b)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) and va.dtype.kind == 'f':
            np.testing.assert_array_equal(va, vb, err_msg=k)
        elif isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=k)
        else:
            assert str(va) == str(vb), k


def _touching_boxes(sess, T):
    """Two slots whose tight boxes overlap on the middle frames and separate at the ends."""
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((2, T, 1, 4), np.float32)
    for t in range(T):
        d = 30.0 if t < 4 or t >= T - 4 else 4.0
        boxes[0, t, 0] = [8, 8, 28, 28]
        boxes[1, t, 0] = [8 + d, 8, 28 + d, 28]
    boxes[..., 0::2] = np.clip(boxes[..., 0::2], 0, w)
    boxes[..., 1::2] = np.clip(boxes[..., 1::2], 0, h)
    return boxes


def test_moe_routes_and_e1_only_is_the_plain_model(tmp_path_factory, capsys):
    _, sess, registry, name = _scene(tmp_path_factory, 'rat', 16)
    torch.manual_seed(0)
    moe = build_model({**BOX, 'set_decoder': True}, n_keypoints=registry.n_keypoints).eval()
    plain = build_model(BOX, n_keypoints=registry.n_keypoints).eval()
    plain.load_state_dict({k: v for k, v in moe.state_dict().items()
                           if not k.startswith('set_decoder.')}, strict=True)
    boxes = _touching_boxes(sess, 16)
    kw = dict(boxes_for=_boxes_for(boxes), n_rows=2)
    cfg = dict(anchor='carry', box_prompt='detector', refine=False)
    base = run_group(plain, sess, 'g000', registry, name, _infer_cfg(**cfg), **kw)
    # a model without E2 ignores the route; E1-only routing on the MoE model is the plain model
    _assert_same(base, run_group(plain, sess, 'g000', registry, name,
                                 _infer_cfg(moe_route='contact', **cfg), **kw))
    _assert_same(base, run_group(moe, sess, 'g000', registry, name,
                                 _infer_cfg(moe_route='e1', **cfg), **kw))
    # a tiny random model can decode the same rows from either expert; shift E2's so its rows show
    moe.set_decoder.decoder.register_forward_hook(
        lambda m, a, o: {**o, 'out_2d': o['out_2d'] + 3.0} if moe.set_decoder._layout else None)
    routed = run_group(moe, sess, 'g000', registry, name, _infer_cfg(moe_route='contact', **cfg), **kw)
    last = [line for line in capsys.readouterr().out.splitlines() if line.startswith('moe ')][-1]
    assert 'route contact' in last and 'E2 on 0 of' not in last
    assert not np.array_equal(np.nan_to_num(routed['pred']), np.nan_to_num(base['pred']))
    # far frames at the clip start come from E1 (identical up to the first contact window)
    np.testing.assert_array_equal(routed['pred'][:, :2], base['pred'][:, :2])
