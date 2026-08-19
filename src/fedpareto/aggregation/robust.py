import numpy as np
import torch

from fedpareto.utils import average_state_dicts, flatten_delta

def _stack_param(client_results, key):
    return torch.stack([c.delta[key].detach().cpu().float() for c in client_results], dim=0)

def coord_median_aggregate(client_results):
    out = {}
    keys = client_results[0].delta.keys()
    for k in keys:
        stacked = _stack_param(client_results, k)
        out[k] = stacked.median(dim=0).values.to(client_results[0].delta[k].device)
    n = len(client_results)
    weights = [1.0 / n] * n
    return out, weights, {"method": "coord_median"}

def trimmed_mean_aggregate(client_results, trim_ratio=0.2):
    out = {}
    keys = client_results[0].delta.keys()
    n = len(client_results)
    trim_k = int(trim_ratio * n)
    for k in keys:
        stacked = _stack_param(client_results, k)
        values, _ = torch.sort(stacked, dim=0)
        trimmed = values[trim_k: n - trim_k] if n - 2 * trim_k > 0 else values
        out[k] = trimmed.mean(dim=0).to(client_results[0].delta[k].device)
    weights = [1.0 / n] * n
    return out, weights, {"method": "trimmed_mean", "trim_ratio": trim_ratio}

def krum_aggregate(client_results, byzantine_count=None):
    vecs = [flatten_delta(c.delta) for c in client_results]
    n = len(vecs)
    f = byzantine_count if byzantine_count is not None else max(0, (n - 3) // 2)
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = torch.norm(vecs[i] - vecs[j]).item() ** 2
            dists.append(d)
        dists.sort()
        score = sum(dists[: max(1, n - f - 2)])
        scores.append(score)
    winner = int(np.argmin(scores))
    out = {k: v.detach().clone() for k, v in client_results[winner].delta.items()}
    weights = [0.0] * n
    weights[winner] = 1.0
    return out, weights, {"method": "krum", "winner_client": client_results[winner].client_id, "scores": scores}
