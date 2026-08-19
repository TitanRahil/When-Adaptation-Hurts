#Below is the **step-by-step way to run the full FedPARETO project** from the zip, with:

---

# 1. Unzip the project

## Command

```bash
unzip FedPARETO_project.zip
cd FedPARETO_project
```

## Input

* `FedPARETO_project.zip`

## Output

A folder named:

```bash
FedPARETO_project/
```

Inside it, you will see:

```bash
configs/
scripts/
src/
runs/
outputs/
README.md
environment.yml
requirements.txt
setup.py
```

---

# 2. Create the environment

## Command

```bash
conda env create -f environment.yml
conda activate fedpareto
pip install -e .
```

## What this does

* creates a Python environment named `fedpareto`
* installs PyTorch, torchvision, numpy, pandas, matplotlib, PyYAML, scikit-learn, tqdm
* installs your project in editable mode

## Input

* `environment.yml`
* local project folder

## Output

A working environment where the command below should work:

```bash
python -c "import fedpareto; print('FedPARETO import successful')"
```

Expected output:

```bash
FedPARETO import successful
```

---

# 3. Check GPU availability

## Command

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Input

* installed PyTorch
* local machine GPU

## Output

Example:

```bash
True
1
NVIDIA L40
```

If it prints `False`, the project will run on CPU only if you change config files from:

```yaml
device: cuda
```

to:

```yaml
device: cpu
```

---

# 4. Understand where datasets go

The code automatically downloads the dataset into:

```bash
data/raw/
```

For example, MNIST will be downloaded under:

```bash
data/raw/MNIST/
```

You do **not** need to manually download MNIST for the provided configs.

---

# 5. First sanity run: FedAvg baseline

This is the best first test because it checks:

* environment
* dataset download
* training loop
* logging
* output saving

## Command

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
```

## Input

The config file:

```bash
configs/experiments/mnist_fedavg.yaml
```

Important values inside it:

* dataset: MNIST
* 20 clients
* 8 clients per round
* 20 FL rounds
* 2 local epochs
* method: `fedavg`
* attack: none

## What happens internally

* MNIST is downloaded if not already present
* training data is partitioned across 20 non-IID clients
* 8 clients are sampled each round
* each selected client trains locally
* server aggregates with FedAvg
* global model is evaluated every round

## Console output

You should see something like:

```bash
Rounds:   5%|█▌                                | 1/20 [00:xx<...]
Rounds: 100%|██████████████████████████████████| 20/20 [..]
Run finished. Outputs saved to: runs/mnist_fedavg_seed1
```

## Output folder created

```bash
runs/mnist_fedavg_seed1/
```

## Files generated

```bash
runs/mnist_fedavg_seed1/config_snapshot.yaml
runs/mnist_fedavg_seed1/metrics_round.csv
runs/mnist_fedavg_seed1/client_weights_round.json
runs/mnist_fedavg_seed1/best_model.pt
runs/mnist_fedavg_seed1/last_model.pt
runs/mnist_fedavg_seed1/summary.json
```

---

# 6. What each output file means

## `config_snapshot.yaml`

The exact configuration used for the run.

## `metrics_round.csv`

Round-by-round values such as:

* test accuracy
* test ECE
* worst-client accuracy
* attack success rate
* weight entropy
* round runtime

Example columns:

```csv
round,test_loss,test_accuracy,test_ece,worst_client_accuracy,attack_success_rate,weight_entropy,round_runtime_sec,num_selected_clients
```

## `client_weights_round.json`

Stores:

* which clients were selected per round
* aggregation weights for those clients
* method diagnostics

## `best_model.pt`

Best global model checkpoint based on highest test accuracy.

## `last_model.pt`

Final checkpoint from the last FL round.

## `summary.json`

Short final summary for paper tables.

Example keys:

```json
{
  "experiment_name": "mnist_fedavg_seed1",
  "method": "fedavg",
  "attack": "none",
  "final_test_accuracy": ...,
  "best_test_accuracy": ...,
  "final_test_ece": ...,
  "final_worst_client_accuracy": ...,
  "final_attack_success_rate": ...,
  "total_runtime_sec": ...
}
```

---

# 7. Run the main proposed method: FedPARETO

## Command

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
```

## Input

The config:

```bash
configs/experiments/mnist_fedpareto.yaml
```

Important method settings:

* method: `fedpareto`
* objective weights:

  * accuracy
  * calibration
  * fairness
  * robustness
* anchor size: 512
* root set used for server reference update

## What happens internally

Compared with FedAvg, this also computes:

* anchor-set summaries per client
* utility-style scores
* fairness-aware term
* temporal reliability
* benign-subspace consistency
* Pareto-rank bonus
* final aggregation weights

## Console output

You should again see the round progress bar and then:

```bash
Run finished. Outputs saved to: runs/mnist_fedpareto_seed1
```

## Output folder

```bash
runs/mnist_fedpareto_seed1/
```

The same core files are generated as in the FedAvg run.

---

# 8. Generate plots for a finished run

Do this after each run you want to visualize.

## Command

```bash
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_seed1
```

## Input

* run directory containing `metrics_round.csv`

## Output

A folder:

```bash
runs/mnist_fedpareto_seed1/plots/
```

Inside it:

```bash
accuracy_vs_round.png
ece_vs_round.png
worst_client_vs_round.png
asr_vs_round.png
weight_entropy_vs_round.png
```

## Meaning of each plot

* `accuracy_vs_round.png` → global test accuracy by FL round
* `ece_vs_round.png` → calibration error by round
* `worst_client_vs_round.png` → fairness-style worst-client performance
* `asr_vs_round.png` → attack success rate for backdoor runs
* `weight_entropy_vs_round.png` → how concentrated or diverse the server weights are

---

# 9. Run poisoning experiments

You already have ready configs for poisoning.

---

## 9.1 Sign-flip poisoning with FedPARETO

### Command

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_signflip.yaml
```

### Input

This config sets:

* method: `fedpareto`
* attack: `sign_flip`
* malicious fraction: `0.25`

This means:

* 25% of total clients are malicious
* malicious clients flip and amplify their updates before sending them to the server

### Output

```bash
runs/mnist_fedpareto_signflip_seed1/
```

---

## 9.2 BadNets backdoor poisoning with FedPARETO

### Command

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_badnets.yaml
```

### Input

This config sets:

* attack: `badnets`
* malicious fraction: `0.25`
* poison fraction: `0.30`
* target label: `0`

This means:

* malicious clients poison a fraction of local mini-batches
* a small patch is added to the bottom-right of the image
* labels are changed to target class `0`

### Output

```bash
runs/mnist_fedpareto_badnets_seed1/
```

### Additional result

Now `attack_success_rate` becomes meaningful and gets logged in:

```bash
metrics_round.csv
summary.json
```

---

# 10. Run other robust baselines

These are already supported.

## FLTrust

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fltrust.yaml
```

Output:

```bash
runs/mnist_fltrust_seed1/
```

## Krum

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_krum.yaml
```

Output:

```bash
runs/mnist_krum_seed1/
```

## Trimmed Mean

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_trimmed_mean.yaml
```

Output:

```bash
runs/mnist_trimmed_mean_seed1/
```

---

# 11. Run all baselines one after another

If you want one command to launch the baseline group:

## Command

```bash
bash scripts/run_all_baselines.sh
```

## What this runs

It sequentially runs:

* FedAvg
* Trimmed Mean
* Krum
* FLTrust
* FedPARETO

## Output

All run folders appear inside:

```bash
runs/
```

---

# 12. Run the ablation suite

This runs several reduced FedPARETO variants.

## Command

```bash
python scripts/run_ablation_suite.py --base_config configs/experiments/mnist_fedpareto.yaml
```

## Input

Base config:

```bash
configs/experiments/mnist_fedpareto.yaml
```

## What it creates

These ablation runs:

* no calibration term
* no fairness term
* no robustness term
* no Pareto bonus

## Output folders

```bash
runs/mnist_fedpareto_seed1_ablation_no_calibration/
runs/mnist_fedpareto_seed1_ablation_no_fairness/
runs/mnist_fedpareto_seed1_ablation_no_robustness/
runs/mnist_fedpareto_seed1_ablation_no_pareto_bonus/
```

Each of them contains:

```bash
config_snapshot.yaml
metrics_round.csv
client_weights_round.json
best_model.pt
last_model.pt
summary.json
```

---

# 13. Create LaTeX tables from all finished runs

After you have completed baselines, attacks, and ablations, export a comparison table.

## Command

```bash
python scripts/export_latex_tables.py --runs_root runs --output outputs/tables/main_results.tex
```

## Input

* all subfolders under `runs/`
* each run must contain a `summary.json`

## Output

Two files:

```bash
outputs/tables/main_results.tex
outputs/tables/main_results.csv
```

## What is inside

A summary table with columns such as:

* experiment name
* method
* attack
* final test accuracy
* best test accuracy
* final ECE
* final worst-client accuracy
* final attack success rate
* total runtime

This can directly go into your paper.

---

# 14. Recommended full run order

This is the safest order to run the project.

## Step A: basic sanity check

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
```

## Step B: main proposed method

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
```

## Step C: robust baselines

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_trimmed_mean.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_krum.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fltrust.yaml
```

## Step D: attack evaluation

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_signflip.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_badnets.yaml
```

## Step E: ablations

```bash
python scripts/run_ablation_suite.py --base_config configs/experiments/mnist_fedpareto.yaml
```

## Step F: plots

```bash
python scripts/make_plots.py --run_dir runs/mnist_fedavg_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_seed1
python scripts/make_plots.py --run_dir runs/mnist_trimmed_mean_seed1
python scripts/make_plots.py --run_dir runs/mnist_krum_seed1
python scripts/make_plots.py --run_dir runs/mnist_fltrust_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_signflip_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_badnets_seed1
```

## Step G: final tables

```bash
python scripts/export_latex_tables.py --runs_root runs --output outputs/tables/main_results.tex
```

---

# 15. How to run on your two servers

You said you have:

* one L40 on one server
* two L40s on another server

The project does not do automatic multi-node FL scheduling, so use **manual experiment distribution**.

## Server 1: single L40

Run lighter experiments:

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_trimmed_mean.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_krum.yaml
```

## Server 2: dual L40s

Use two terminals, one per GPU.

### Terminal 1

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
```

### Terminal 2

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_badnets.yaml
```

Then continue with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/experiments/mnist_fltrust.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation_suite.py --base_config configs/experiments/mnist_fedpareto.yaml
```

This is the most practical use of your setup.

---

# 16. How to create multiple seeds

Right now the provided configs are all `seed: 1`.

For journal-quality results, create seed variants such as:

* seed 1
* seed 2
* seed 3

Example:
copy

```bash
configs/experiments/mnist_fedpareto.yaml
```

to

```bash
configs/experiments/mnist_fedpareto_seed2.yaml
configs/experiments/mnist_fedpareto_seed3.yaml
```

Then change:

```yaml
experiment_name: mnist_fedpareto_seed2
seed: 2
```

and

```yaml
experiment_name: mnist_fedpareto_seed3
seed: 3
```

Run them:

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_seed2.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_seed3.yaml
```

Do the same for baselines.

---

# 17. Common input settings you may want to change

Inside each YAML config:

## Change dataset

```yaml
dataset:
  name: mnist
```

Can be changed to:

```yaml
dataset:
  name: fashionmnist
```

or

```yaml
dataset:
  name: cifar10
```

## Change number of rounds

```yaml
federated:
  rounds: 20
```

## Change clients per round

```yaml
federated:
  clients_per_round: 8
```

## Change method

```yaml
method:
  name: fedpareto
```

## Change attack

```yaml
attack:
  name: none
```

or

```yaml
attack:
  name: sign_flip
```

or

```yaml
attack:
  name: badnets
```

---

# 18. Final folders you should have after a full study

After everything finishes, you should see:

```bash
runs/
├── mnist_fedavg_seed1/
├── mnist_fedpareto_seed1/
├── mnist_trimmed_mean_seed1/
├── mnist_krum_seed1/
├── mnist_fltrust_seed1/
├── mnist_fedpareto_signflip_seed1/
├── mnist_fedpareto_badnets_seed1/
├── mnist_fedpareto_seed1_ablation_no_calibration/
├── mnist_fedpareto_seed1_ablation_no_fairness/
├── mnist_fedpareto_seed1_ablation_no_robustness/
└── mnist_fedpareto_seed1_ablation_no_pareto_bonus/
```

And:

```bash
outputs/tables/main_results.tex
outputs/tables/main_results.csv
```

And inside each run:

```bash
plots/
```

with PNG graphs.

---

# 19. Shortest command list for the entire project

If you just want the minimal complete run sequence:

```bash
conda env create -f environment.yml
conda activate fedpareto
pip install -e .

python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_trimmed_mean.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_krum.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fltrust.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_signflip.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto_badnets.yaml
python scripts/run_ablation_suite.py --base_config configs/experiments/mnist_fedpareto.yaml

python scripts/make_plots.py --run_dir runs/mnist_fedavg_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_seed1
python scripts/make_plots.py --run_dir runs/mnist_trimmed_mean_seed1
python scripts/make_plots.py --run_dir runs/mnist_krum_seed1
python scripts/make_plots.py --run_dir runs/mnist_fltrust_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_signflip_seed1
python scripts/make_plots.py --run_dir runs/mnist_fedpareto_badnets_seed1

python scripts/export_latex_tables.py --runs_root runs --output outputs/tables/main_results.tex
```

---

# 20. Important note

This project is a **research scaffold**. It will generate:

* results
* CSV logs
* plots
* tables
* checkpoints

But it will **not magically produce journal-quality final conclusions in one run**.
For a real paper, you should run:

* multiple seeds
* multiple datasets
* multiple attack strengths
* multiple heterogeneity levels

