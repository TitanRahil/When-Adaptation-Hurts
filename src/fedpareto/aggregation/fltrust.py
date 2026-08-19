import numpy as np
import torch

from fedpareto.utils import average_state_dicts, flatten_delta, cosine_similarity

def fltrust_aggregate(client_results, root_delta):
    root_vec = flatten_delta(root_delta)
    trust_scores = []
    normalized_deltas = []

    root_norm = root_vec.norm().item() + 1e-12
    for c in client_results:
        vec = flatten_delta(c.delta)
        cos = max(0.0, cosine_similarity(vec, root_vec))
        trust_scores.append(cos)
        vec_norm = vec.norm().item() + 1e-12
        scale = root_norm / vec_norm
        normalized_deltas.append({k: v * scale for k, v in c.delta.items()})

    scores = np.asarray(trust_scores, dtype=np.float64)
    if scores.sum() <= 0:
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = scores / scores.sum()
    agg = average_state_dicts(normalized_deltas, weights)
    return agg, weights.tolist(), {"method": "fltrust", "trust_scores": trust_scores}
