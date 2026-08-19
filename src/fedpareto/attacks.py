from typing import Dict
import torch


def apply_model_attack(delta: Dict[str, torch.Tensor], attack_cfg):
    name = attack_cfg["name"]

    if name == "none":
        return delta

    if name == "sign_flip":
        out = {}
        for k, v in delta.items():
            if v.is_floating_point():
                out[k] = -3.0 * v
            else:
                out[k] = v.detach().clone()
        return out

    if name == "gaussian":
        out = {}
        std = float(attack_cfg.get("std", 0.5))
        for k, v in delta.items():
            if v.is_floating_point():
                out[k] = v + torch.randn_like(v) * std
            else:
                out[k] = v.detach().clone()
        return out

    return delta


def maybe_poison_batch(x, y, attack_cfg, malicious: bool):
    if not malicious:
        return x, y

    if attack_cfg["name"] != "badnets":
        return x, y

    poison_fraction = float(attack_cfg.get("poison_fraction", 0.30))
    target_label = int(attack_cfg.get("target_label", 0))
    patch_value = float(attack_cfg.get("patch_value", 1.0))
    trigger_size = int(attack_cfg.get("trigger_size", max(2, x.size(-1) // 8)))

    batch_size = x.size(0)
    poison_count = max(1, int(poison_fraction * batch_size))

    x = x.clone()
    y = y.clone()
    x[:poison_count, :, -trigger_size:, -trigger_size:] = patch_value
    y[:poison_count] = target_label
    return x, y