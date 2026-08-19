from dataclasses import dataclass
from typing import Dict

import torch


def _clean_singleton_safe_forward(model, x, cfg):
    """Safely forward singleton local mini-batches through BatchNorm models.

    A one-sample local batch can produce a [1, C, 1, 1] activation, for which
    BatchNorm cannot compute training statistics. For singleton batches only,
    BatchNorm layers temporarily use their stored running statistics while the
    remainder of the network stays in training mode.

    This rule is applied consistently to clean and attacked configurations.
    """
    if x.shape[0] != 1:
        return model(x)

    import torch

    bn_layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]

    if not bn_layers:
        return model(x)

    training_states = [module.training for module in bn_layers]

    try:
        for module in bn_layers:
            module.eval()

        return model(x)

    finally:
        for module, was_training in zip(bn_layers, training_states):
            module.train(was_training)


import torch.nn.functional as F

from fedpareto.attacks import apply_model_attack, maybe_poison_batch
from fedpareto.anchor import summarize_on_anchor
from fedpareto.metrics import evaluate_model
from fedpareto.utils import clone_state_dict, subtract_state_dicts

@dataclass
class ClientResult:
    client_id: int
    samples: int
    delta: Dict[str, torch.Tensor]
    anchor_summary: Dict[str, torch.Tensor]
    local_metrics: Dict[str, float]
    is_malicious: bool

class LocalClient:
    def __init__(self, client_id, model_fn, train_loader, eval_loader, device, cfg, malicious=False):
        self.client_id = client_id
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.cfg = cfg
        self.malicious = malicious

    def train(self, global_state, anchor_loader):
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_state, strict=True)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.cfg["federated"]["lr"],
            momentum=self.cfg["federated"]["momentum"],
            weight_decay=self.cfg["federated"]["weight_decay"],
        )

        local_epochs = self.cfg["federated"]["local_epochs"]
        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = maybe_poison_batch(x, y, self.cfg["attack"], self.malicious)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = _clean_singleton_safe_forward(model, x, self.cfg)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

        new_state = clone_state_dict(model.state_dict())
        delta = subtract_state_dicts(new_state, global_state)
        delta = apply_model_attack(delta, self.cfg["attack"] if self.malicious else {"name": "none"})

        local_metrics = evaluate_model(
            model,
            self.eval_loader,
            self.device,
            ece_bins=self.cfg["evaluation"]["ece_bins"],
        )
        anchor_summary = summarize_on_anchor(model, anchor_loader, self.device)
        return ClientResult(
            client_id=self.client_id,
            samples=len(self.train_loader.dataset),
            delta=delta,
            anchor_summary=anchor_summary,
            local_metrics=local_metrics,
            is_malicious=self.malicious,
        )
