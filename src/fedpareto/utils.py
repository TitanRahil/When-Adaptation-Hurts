import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    vecs = [
        v.detach().reshape(-1).float().cpu()
        for v in state_dict.values()
        if v.is_floating_point()
    ]
    return torch.cat(vecs) if vecs else torch.zeros(1)


def flatten_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    vecs = [
        v.detach().reshape(-1).float().cpu()
        for v in delta.values()
        if v.is_floating_point()
    ]
    return torch.cat(vecs) if vecs else torch.zeros(1)


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def subtract_state_dicts(new_state, old_state):
    out = {}
    for k in new_state.keys():
        if new_state[k].is_floating_point():
            out[k] = new_state[k].detach().clone() - old_state[k].detach().clone()
        else:
            out[k] = torch.zeros_like(new_state[k])
    return out


def add_delta_to_state(base_state, delta, scale=1.0):
    out = {}
    for k in base_state.keys():
        if base_state[k].is_floating_point():
            out[k] = base_state[k].detach().clone() + scale * delta[k].detach().clone().to(base_state[k].dtype)
        else:
            out[k] = base_state[k].detach().clone()
    return out


def average_state_dicts(deltas, weights):
    out = {}
    keys = deltas[0].keys()
    for k in keys:
        ref = deltas[0][k]
        if ref.is_floating_point():
            agg = torch.zeros_like(ref)
            for d, w in zip(deltas, weights):
                agg.add_(d[k].to(agg.dtype), alpha=float(w))
            out[k] = agg
        else:
            out[k] = torch.zeros_like(ref)
    return out


def apply_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def entropy_from_weights(weights: List[float]) -> float:
    eps = 1e-12
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, eps, 1.0)
    return float(-(w * np.log(w)).sum())


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    denom = (x.norm() * y.norm()).item() + 1e-12
    return float(torch.dot(x, y).item() / denom)


def project_simplex(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D")
    if scores.sum() == 1.0 and np.all(scores >= 0):
        return scores
    u = np.sort(scores)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(scores - theta, 0)
    if w.sum() <= 0:
        w = np.ones_like(scores) / len(scores)
    else:
        w = w / w.sum()
    return w