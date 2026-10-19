"""Under Muon the fresh non-matrix params train at `kpt_lr`, as under AdamW-SF (model_train B5)."""
import pytest
import torch
import torch.nn as nn

from tailcyclenet.pose.optim import ROUTES, build_muon, build_optimizer, group_lr, route_param


class _Fresh(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mlps = nn.ModuleList([nn.Linear(8, 8)])
        self.query_encoder = nn.Module()
        self.query_encoder.kpt_embed = nn.Embedding(5, 8)
        self.query_encoder.missing_qpos = nn.Parameter(torch.zeros(8))
        self.head = nn.Linear(8, 3)


CFG = dict(learning_rate=1e-4, kpt_lr=5e-4, weight_decay=0.0)


def _lr_of(opt, p):
    for o in opt._opts if hasattr(opt, '_opts') else [opt]:
        for g in o.param_groups:
            if any(q is p for q in g['params']):
                return g['lr']
    raise AssertionError('param not routed')


@pytest.mark.parametrize('kind', ['muon', 'schedulefree'])
def test_identity_table_and_missing_tokens_train_at_kpt_lr(kind):
    m = _Fresh()
    fresh = {'query_encoder.missing_qpos'}
    opt = build_optimizer(m, fresh, dict(CFG, optimizer=kind))
    assert _lr_of(opt, m.query_encoder.kpt_embed.weight) == pytest.approx(5e-4)
    assert _lr_of(opt, m.query_encoder.missing_qpos) == pytest.approx(5e-4)
    assert _lr_of(opt, m.head.bias) == pytest.approx(1e-4)


def test_adamw_fresh_route_and_its_build_order():
    m = _Fresh()
    embed = {id(m.query_encoder.kpt_embed.weight)}
    assert route_param('query_encoder.kpt_embed.weight', m.query_encoder.kpt_embed.weight,
                       set(), embed) == 'adamw_fresh'
    assert ROUTES == ('muon_dec', 'muon_fresh', 'muon_enc', 'adamw_fresh', 'adamw_base',
                      'adamw_enc')
    assert group_lr('adamw_fresh', CFG) == pytest.approx(5e-4)
    opt = build_muon(m, {'query_encoder.missing_qpos'}, CFG)
    assert [g['lr'] for g in opt.opt_adam.param_groups] == pytest.approx([5e-4, 1e-4])
