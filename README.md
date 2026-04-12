# MiSum Reproduction

PyTorch reproduction of the FSE 2025 paper **"MiSum: Multi-modality Heterogeneous Code Graph Learning for Multi-intent Binary Code Summarization"**.

This repo currently supports:

- single-run training and evaluation
- full-data training for one dataset split
- the complete 9-setting experiment matrix:
  - `ARM / X86 / X64`
  - `O1 / O2 / O3`
- automatic result collection into a single summary table

## Repository layout

```text
.
|-- configs/
|   |-- arm_o1_smoke.yaml
|   |-- arm_o1_full.yaml
|   `-- full_runs/
|       |-- arm_o1_full.yaml
|       |-- arm_o2_full.yaml
|       |-- arm_o3_full.yaml
|       |-- x86_o1_full.yaml
|       |-- x86_o2_full.yaml
|       |-- x86_o3_full.yaml
|       |-- x64_o1_full.yaml
|       |-- x64_o2_full.yaml
|       `-- x64_o3_full.yaml
|-- datasets/
|-- misum_repro/
|   |-- data.py
|   |-- metrics.py
|   |-- model.py
|   |-- tokenization.py
|   |-- utils.py
|   `-- vocab.py
|-- artifacts/
|-- train.py
`-- run_experiments.py
```

## Environment

Tested with:

- Windows
- Python `3.14`
- PyTorch `2.11.0+cu128`
- CUDA available

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify your CUDA PyTorch installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Expected dataset layout

Each dataset directory should look like this:

```text
datasets/dataset_gcc-7.3.0_arm_32_O1_strip/
|-- train.jsonl
|-- valid.jsonl
`-- test.jsonl
```

If `valid.jsonl` is not present, the loader automatically falls back to:

- `misum_like_valid.jsonl`

Supported dataset roots in this workspace:

- `datasets/dataset_gcc-7.3.0_arm_32_O1_strip`
- `datasets/dataset_gcc-7.3.0_arm_32_O2_strip`
- `datasets/dataset_gcc-7.3.0_arm_32_O3_strip`
- `datasets/dataset_gcc-7.3.0_x86_32_O1_strip`
- `datasets/dataset_gcc-7.3.0_x86_32_O2_strip`
- `datasets/dataset_gcc-7.3.0_x86_32_O3_strip`
- `datasets/dataset_gcc-7.3.0_x86_64_O1_strip`
- `datasets/dataset_gcc-7.3.0_x86_64_O2_strip`
- `datasets/dataset_gcc-7.3.0_x86_64_O3_strip`

## Quick start

### 1. Smoke run

Use a small subset first to make sure the training loop works:

```powershell
python train.py --config configs/arm_o1_smoke.yaml
```

This is the fastest way to validate:

- dataset loading
- model forward/backward
- evaluation
- artifact generation

### 2. Single full run

Run the full `ARM O1` setting:

```powershell
python train.py --config configs/full_runs/arm_o1_full.yaml
```

### 3. Full 9-setting matrix

Run all architecture / optimization combinations:

```powershell
python run_experiments.py
```

Resume unfinished experiments and skip completed ones:

```powershell
python run_experiments.py --skip-existing
```

Run only a subset:

```powershell
python run_experiments.py --experiments arm_o1_full x64_o3_full
```

Preview commands without starting training:

```powershell
python run_experiments.py --dry-run
```

Stop immediately if one experiment fails:

```powershell
python run_experiments.py --stop-on-error
```

## Scripts

### `train.py`

Main training and evaluation entry point.

Responsibilities:

- load dataset splits
- build source / target vocabularies
- instantiate the MiSum-style model
- train with optional AMP
- evaluate on validation and test
- save checkpoints and prediction files

Usage:

```powershell
python train.py --config <path-to-yaml>
```

### `run_experiments.py`

Batch runner for the complete experiment matrix.

Responsibilities:

- launch multiple `train.py` runs
- support subset selection
- skip completed runs
- collect final metrics into one summary file

Outputs:

- `artifacts/experiment_matrix/summary.json`
- `artifacts/experiment_matrix/summary.csv`

## Configuration guide

All runs are controlled by YAML config files.

### `dataset`

Controls the dataset path and truncation limits.

Example:

```yaml
dataset:
  root: datasets/dataset_gcc-7.3.0_arm_32_O1_strip
  max_train_samples: null
  max_valid_samples: null
  max_test_samples: null
  max_pseudo_nodes: 256
  max_asm_blocks: 128
  max_node_tokens: 32
  max_target_tokens: 32
```

What matters most:

- `root`: dataset directory
- `max_*_samples`: useful for debugging and smoke runs
- `max_pseudo_nodes`: cap for pseudo-code graph size
- `max_asm_blocks`: cap for assembly CFG size
- `max_node_tokens`: token cap per graph node
- `max_target_tokens`: token cap for generated summaries

### `vocab`

Controls vocabulary pruning.

```yaml
vocab:
  src_min_freq: 2
  tgt_min_freq: 1
  src_max_size: 40000
  tgt_max_size: 12000
```

### `model`

Controls encoder / decoder size.

```yaml
model:
  d_model: 256
  num_heads: 4
  num_layers: 4
  dropout: 0.2
  ff_multiplier: 4
  max_positions: 768
```

### `dataloader`

Controls worker processes and host-to-device transfer settings.

```yaml
dataloader:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

### `train`

Controls optimization and GPU behavior.

```yaml
train:
  batch_size: 4
  eval_batch_size: 4
  epochs: 10
  learning_rate: 0.0003
  weight_decay: 0.00001
  grad_clip: 1.0
  early_stopping_patience: 4
  gradient_accumulation_steps: 8
  use_amp: true
  amp_dtype: float16
  allow_tf32: true
  matmul_precision: high
```

Recommended tuning order if you hit memory limits:

1. reduce `batch_size`
2. reduce `max_pseudo_nodes`
3. reduce `max_asm_blocks`
4. reduce `d_model`
5. increase `gradient_accumulation_steps` to recover effective batch size

### `decode`

Controls generation length.

```yaml
decode:
  max_new_tokens: 32
```

## Outputs

Each run writes to:

```text
artifacts/<run_name>/
```

Typical contents:

- `best.pt`
- `config.yaml`
- `src_vocab.json`
- `tgt_vocab.json`
- `metrics.json`
- `predictions_valid.jsonl`
- `predictions_test.jsonl`

`metrics.json` contains:

- training history by epoch
- best validation metrics
- final test metrics
- parameter count
- device info
- AMP flag
- dataset sizes

## Example result

One completed full run in this workspace:

- setting: `ARM O1`
- run name: `arm_o1_full`
- best valid BLEU-4: `1.3384`
- test BLEU-4: `1.2495`

The exact numbers can be found in:

- `artifacts/arm_o1_full/metrics.json`

## Limitations

- this is a reproduction, not the official author code
- alignment edges are heuristic when not present in the dataset
- current metrics are for the local dataset provided in this workspace
- results are not expected to exactly match the paper

## Citation

If you use this repository, please cite the original paper:

```bibtex
@article{zhu2025misum,
  title={MiSum: Multi-modality Heterogeneous Code Graph Learning for Multi-intent Binary Code Summarization},
  author={Zhu, Kangchen and Tian, Zhiliang and Wang, Shangwen and Chen, Weiguo and Dong, Zixuan and Leng, Mingyue and Mao, Xiaoguang},
  journal={Proceedings of the ACM on Software Engineering},
  volume={2},
  number={FSE},
  year={2025},
  doi={10.1145/3715780}
}
```
