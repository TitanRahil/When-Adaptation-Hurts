# FedPARETO

**FedPARETO: Multi-Objective Utility-Guided Aggregation for Federated Learning under Heterogeneity and Poisoning**

This repository contains a practical, runnable research scaffold for the FedPARETO project. It supports:

- federated training on image datasets such as MNIST, FashionMNIST, and CIFAR-10
- non-IID client partitioning using Dirichlet or pathological class skew
- malicious client simulation with sign-flip, Gaussian model poisoning, and BadNets-style backdoor poisoning
- server-side aggregation with **FedAvg**, **coordinate median**, **trimmed mean**, **Krum**, **FLTrust**, and **FedPARETO**
- per-round tracking of:
  - global test accuracy
  - expected calibration error (ECE)
  - worst-client accuracy
  - attack success rate (for backdoor experiments)
  - server aggregation weights
- automatic export of:
  - round-wise CSV logs
  - summary JSON files
  - paper-ready plots
  - LaTeX tables

> The code is written as a **research implementation**. It is meant to be understandable, reproducible, and easy to extend. It is not a highly optimized production FL framework.

## 1. Quick start

```bash
conda env create -f environment.yml
conda activate fedpareto
pip install -e .
```

Run a baseline:

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
```

Run FedPARETO:

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
```

Create plots for a finished run:

```bash
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_seed1
```

Export a LaTeX comparison table from multiple runs:

```bash
python scripts/export_latex_tables.py --runs_root runs --output outputs/tables/main_results.tex
```

## 2. Repository layout

- `configs/`: experiment YAML files
- `scripts/`: command-line entry points
- `src/fedpareto/`: core framework
- `runs/`: per-run outputs
- `outputs/`: plots and tables aggregated across runs
- `docs/diagrams/`: architecture and semantic-flow diagrams

## 3. Implemented methods

### Aggregation baselines
- FedAvg
- Coordinate-wise median
- Trimmed mean
- Krum
- FLTrust

### Proposed method
- FedPARETO

FedPARETO computes client weights from:
- anchor-set accuracy gain
- anchor-set calibration gain
- fairness-aware priority for under-served clients
- temporal reliability
- benign-subspace consistency
- Pareto-rank bonus

## 4. Main outputs

Each run directory contains:
- `config_snapshot.yaml`
- `metrics_round.csv`
- `client_weights_round.json`
- `summary.json`
- `best_model.pt`
- `last_model.pt`

Plot scripts produce:
- `accuracy_vs_round.png`
- `ece_vs_round.png`
- `worst_client_vs_round.png`
- `asr_vs_round.png`
- `weight_entropy_vs_round.png`

## 5. Suggested workflow

1. Start with `mnist_fedavg.yaml`
2. Run `mnist_fedpareto.yaml`
3. Run poisoning experiments with BadNets or sign-flip
4. Produce comparison plots and LaTeX tables
5. Extend to CIFAR-10 or FashionMNIST

## 6. Notes on practicality

This project is intentionally designed to fit a modest research setup such as:
- one L40 GPU on one server
- two L40 GPUs on another server

The code supports single-process GPU execution, while repeated seeds, baselines, and attack studies can be distributed manually across machines.
