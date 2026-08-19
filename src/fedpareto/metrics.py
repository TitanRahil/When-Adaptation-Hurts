from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = accuracies[in_bin].float().mean()
            conf = confidences[in_bin].mean()
            ece += torch.abs(conf - acc) * prop
    return float(ece.item())

@torch.no_grad()
def evaluate_model(model, loader, device, ece_bins=15):
    model.eval()
    total = 0
    correct = 0
    losses = []
    probs_all = []
    labels_all = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()
        losses.append(loss.item() * y.size(0))
        probs_all.append(probs)
        labels_all.append(y)
    probs_all = torch.cat(probs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    acc = correct / max(1, total)
    avg_loss = sum(losses) / max(1, total)
    ece = expected_calibration_error(probs_all, labels_all, n_bins=ece_bins)
    return {"loss": avg_loss, "accuracy": acc, "ece": ece}

@torch.no_grad()
def worst_client_accuracy(model, client_eval_loaders, device, ece_bins=15):
    accs = []
    metrics = {}
    for cid, loader in client_eval_loaders.items():
        out = evaluate_model(model, loader, device, ece_bins=ece_bins)
        accs.append(out["accuracy"])
        metrics[cid] = out
    return (min(accs) if accs else 0.0), metrics

@torch.no_grad()
def attack_success_rate(model, loader, device, target_label=0, patch_value=1.0, trigger_size=4):
    model.eval()
    total = 0
    success = 0
    for x, y in loader:
        x = x.clone()
        x[:, :, -trigger_size:, -trigger_size:] = patch_value
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        total += y.size(0)
        success += (preds == target_label).sum().item()
    return success / max(1, total)
