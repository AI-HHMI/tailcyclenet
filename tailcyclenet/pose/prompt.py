"""The label-free self-prompted regime: re-query at the model's own frame-0 prediction."""
import torch


def self_prompt(model, views, kpt_ids, cgroup, mode, first, kpt_chunk=None, box_prompt=None):
    """Re-query at the model's own frame-0 prediction -- the label-free prompted regime.

    `first` is a completed prior-free pass; its frame-0 pose becomes the prior for a second pass,
    already in the model's own frame (the pose is `(1,K,R)`). It gets the same bounds mask
    `carry` gets: a keypoint outside its own crop is NaN, which is what the no-query tokens key
    off. The box prompt carries into the second pass unchanged: the window has not moved. Shared
    with the trainer so training and inference report the same number. `prior_out_of_bounds` is
    imported here, not at module level: `dataset` pulls pandas through tracktail, and table-only
    modules that import the model must not.
    """
    from ..data.camera import prior_out_of_bounds

    p = first['coords_pred'][0].detach()
    prior = p[0][None].clone()
    prior[0, prior_out_of_bounds(prior[0], mode, cgroup)] = float('nan')
    qt = torch.zeros(prior.shape[:2], dtype=torch.int32, device=prior.device)
    mkw = {} if box_prompt is None else {'box_prompt': box_prompt}
    return model(views, kpt_ids, cgroup, mode=mode, kpt_prior=prior, prompt_time=qt,
                 kpt_chunk=kpt_chunk, **mkw)
