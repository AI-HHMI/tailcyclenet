"""`[training.speed]`: the option table, its refusals, and what it patches on a MoE model."""
import pytest
import torch

from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.speed import DEFAULTS, apply_training_speedups, decoder_blocks, speed_options
from tests.helpers import SMALL

BOX = {**SMALL, 'box_prompt': 'film'}


def test_defaults_and_off_switch():
    assert speed_options({}) == {**DEFAULTS}
    off = speed_options({'speed': {'enabled': False}})
    assert off['autocast_bf16'] == [] and off['compile'] == []
    assert off['encoder_activation_checkpointing'] == DEFAULTS['encoder_activation_checkpointing']
    assert speed_options({'speed': {'compile': []}})['autocast_bf16'] == ['encoder', 'decoder']


@pytest.mark.parametrize('section', [{'compile_mode': 'x'}, {'autocast_bf16': ['heads']},
                                     {'compile': ['decoder']}])
def test_unknown_keys_and_parts_raise(section):
    with pytest.raises(SystemExit):
        speed_options({'speed': section})


def test_moe_lists_both_decoders_and_cpu_skips_by_name():
    torch.manual_seed(0)
    moe = build_model({**BOX, 'set_decoder': True}, n_keypoints=5)
    plain = build_model(BOX, n_keypoints=5)
    assert len(decoder_blocks(moe)) == 2 * len(decoder_blocks(plain)) > 0
    keys = set(moe.state_dict())
    info = apply_training_speedups(moe, {'speed': {'encoder_activation_checkpointing': False}},
                                   'cpu')
    assert info['autocast_bf16'] == [] and info['compile'] == []
    assert 'autocast_bf16_skipped' in info and 'compile_skipped' in info
    assert moe.scene_encoder.encoder.use_activation_checkpointing is False
    assert info['encoder_activation_checkpointing'] is False
    assert set(moe.state_dict()) == keys


def test_selective_checkpoint_matches_eager_and_skips_frozen_blocks(monkeypatch):
    """Checkpointing only trainable blocks gives the same grads as no patch, and a frozen block
    with a grad-free input is never checkpointed."""
    from tailcyclenet.pose import speed as S
    torch.manual_seed(0)
    blk = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU())
    calls = []
    real = S.checkpoint
    monkeypatch.setattr(S, 'checkpoint', lambda *a, **k: calls.append(1) or real(*a, **k))
    x = torch.randn(3, 4)
    blk.requires_grad_(False)
    assert calls == [] and not S._block_forward(blk, blk.forward, True, x).requires_grad
    blk.requires_grad_(True)
    blk(x).sum().backward()
    ref = [p.grad.clone() for p in blk.parameters()]
    blk.zero_grad()
    S._block_forward(blk, blk.forward, True, x).sum().backward()
    assert calls == [1]
    for p, g in zip(blk.parameters(), ref):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_patches_keep_state_dict_keys_on_cuda():
    torch.manual_seed(0)
    moe = build_model({**BOX, 'set_decoder': True}, n_keypoints=5).cuda()
    keys = set(moe.state_dict())
    info = apply_training_speedups(moe, {'speed': {'compile': []}}, 'cuda')
    assert info['autocast_bf16'] == ['decoder', 'encoder']
    assert set(moe.state_dict()) == keys
