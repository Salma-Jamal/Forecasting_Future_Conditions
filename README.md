# EHR Future Conditions Forecasting

A PyTorch pipeline for predicting incident clinical conditions from longitudinal electronic health record (EHR) events. The project converts timestamped patient histories into fixed-length sequences, optionally pretrains a causal Transformer with next-event prediction, and fine-tunes it as a multi-label classifier over a configurable set of target condition codes.

The pipeline supports seven event types, demographic embeddings, continuous time features, class-imbalance-aware loss, multi-anchor training augmentation, early stopping, optional mixed precision on CUDA, and optional Weights & Biases logging.

## How it works

For each patient, the pipeline:

1. Chooses an **anchor date**.
2. Collects coded clinical events occurring on or before that date.
3. Sorts the events chronologically and retains the most recent `max_seq_len` events.
4. Represents every event with code, event type, position, time, gender, and race embeddings/features.
5. Predicts which target conditions will first occur during the five years after the anchor.

A condition is positive only when it occurs in the prediction window and was not recorded before the anchor. Training and validation anchors are five years before each patient's last encounter. Test anchors come from `test_anchors.csv` when supplied; otherwise, they are derived the same way.

The optional pretraining stage learns next-event code prediction with a causal attention mask. Fine-tuning uses last-token pooling and a multi-label classification head. By default, it combines focal classification loss with an auxiliary language-modeling loss.

## Repository layout

| File | Purpose |
| --- | --- |
| `run.py` | Command-line entry point and end-to-end orchestration |
| `config.py` | Model, data, training, loss, and augmentation defaults |
| `data.py` | CSV loading, anchors, labels, vocabulary, sequences, and augmentation |
| `dataset.py` | PyTorch datasets for pretraining and classification |
| `model.py` | Causal Transformer decoder and classification head |
| `trainer.py` | Losses, optimization, metrics, checkpoints, and prediction |

## Requirements

- Python 3.10 or newer is recommended
- PyTorch
- pandas
- NumPy
- scikit-learn
- tqdm
- Weights & Biases (`wandb`), optional

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pandas numpy scikit-learn tqdm
```

For experiment tracking, also install:

```bash
python -m pip install wandb
```

Install the appropriate PyTorch build for your platform if you require CUDA support. See the official PyTorch installation instructions for platform-specific commands.

## Data layout

Pass a data directory with this structure:

```text
data/
├── patient_splits.csv
├── target_conditions.csv
├── test_anchors.csv              # optional
├── train_val/
│   ├── patients.csv
│   ├── encounters.csv
│   ├── conditions.csv
│   ├── observations.csv
│   ├── medications.csv
│   ├── procedures.csv
│   ├── allergies.csv             # optional/loaded but not tokenized
│   ├── immunizations.csv
│   ├── careplans.csv
│   ├── devices.csv               # optional/loaded but not tokenized
│   └── imaging_studies.csv       # optional/loaded but not tokenized
└── test/
    └── ...same table names...
```

Missing clinical tables are skipped. In practice, `patients.csv`, `encounters.csv`, and `conditions.csv` are needed for demographics, anchor construction, and labels respectively.

### Metadata files

`patient_splits.csv` must contain:

| Column | Description |
| --- | --- |
| `Id` | Patient identifier matching IDs in the clinical tables |
| `split` | One of `train`, `val`, or `test` |

Example:

```csv
Id,split
patient-001,train
patient-002,val
patient-003,test
```

`target_conditions.csv` must contain a `CODE` column. Each row defines one output label and one column in `predictions.csv`.

```csv
CODE
44054006
195967001
```

Optional `test_anchors.csv` must contain:

```csv
Id,anchor_date
patient-003,2020-01-01
```

## Quick start

Run the complete pretraining and fine-tuning pipeline:

```bash
python run.py \
  --data-dir ./data \
  --output-dir ./outputs/ehr-transformer \
  --device cpu
```

On a CUDA machine, omit `--device` to select CUDA automatically or set it explicitly:

```bash
python run.py --data-dir ./data --output-dir ./outputs/run-1 --device cuda
```

For a short test that skips pretraining:

```bash
python run.py \
  --data-dir ./data \
  --output-dir ./outputs/short_test \
  --skip-pretrain \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --max-seq-len 64
```

Training without pretraining initializes the decoder randomly.

### Reuse a pretrained decoder

Use both flags to bypass pretraining and load an existing decoder checkpoint:

```bash
python run.py \
  --data-dir ./data \
  --output-dir ./outputs/finetune \
  --skip-pretrain \
  --pretrain-ckpt ./outputs/pretraining/pretrained_decoder.pt
```

The checkpoint architecture must match the current vocabulary size, model width, layer count, and related model settings. The vocabulary is rebuilt from the current training split, so checkpoints are safest to reuse with the same data and configuration.

### Weights & Biases

```bash
wandb login
python run.py \
  --data-dir ./data \
  --output-dir ./outputs/tracked-run \
  --wandb \
  --wandb-project ehr-transformer \
  --wandb-run-name baseline
```

If `wandb` is unavailable, the pipeline prints a warning and continues without logging.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `/workspace/transf/data` | Input data directory |
| `--output-dir` | `/workspace/transf/output_transformer_decoder` | Checkpoint and prediction directory |
| `--batch-size` | `32` | Fine-tuning batch size |
| `--epochs` | `60` | Maximum fine-tuning epochs |
| `--lr` | `1e-4` | Classification-head learning rate |
| `--d-model` | `256` | Transformer hidden size |
| `--nhead` | `8` | Attention heads; must divide `d-model` |
| `--num-layers` | `4` | Transformer layers |
| `--max-seq-len` | `512` | Maximum events retained per patient |
| `--device` | automatic | PyTorch device, such as `cpu`, `cuda`, or `mps` |
| `--pretrain-ckpt` | none | Decoder state-dict checkpoint to load for fine-tuning |
| `--skip-pretrain` | false | Skip next-event pretraining |
| `--wandb` | false | Enable W&B logging |
| `--wandb-project` | `ehr-transformer` | W&B project name |
| `--wandb-run-name` | generated | W&B run name |

Run `python run.py --help` for the authoritative CLI reference.

## Configuration

Settings not exposed through the CLI are defined in the `Config` dataclass in `config.py`. Important defaults include:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `min_code_freq` | 3 | Minimum training frequency for vocabulary inclusion |
| `dim_feedforward` | 1024 | Transformer feed-forward width |
| `dropout` | 0.15 | Decoder dropout |
| `pretrain_epochs` | 50 | Maximum pretraining epochs |
| `pretrain_batch_size` | 64 | Pretraining batch size |
| `patience` | 10 | Fine-tuning early-stopping patience |
| `pretrain_patience` | 10 | Pretraining early-stopping patience |
| `encoder_lr_scale` | 0.1 | Decoder LR relative to the classification head |
| `use_focal_loss` | true | Use focal loss instead of weighted BCE |
| `focal_gamma` | 1.5 | Focal-loss focusing parameter |
| `aux_lm_weight` | 0.1 | Next-event loss weight during fine-tuning |
| `do_augment` | true | Enable alternative-anchor training examples |
| `augment_n_per_patient` | 2 | Candidate augmented anchors per training patient |
| `num_workers` | 4 | PyTorch data-loader workers |

To change these values, edit `config.py` or expose additional fields as CLI arguments in `run.py`.

## Sequence representation

The vocabulary is fitted only on events belonging to training patients, includes codes occurring at least `min_code_freq` times, and is capped at 10,000 entries. Rare or unseen codes map to the `[MASK]` token.

Each event receives three normalized time features:

- Days before the anchor, clipped at `max_time_days` (10 years by default)
- Patient age at the event, clipped at 100 years
- Days since the preceding event, clipped at one year

The model adds token, event-type, position, projected-time, gender, and race representations. Sequences are chronological, left-aligned, right-padded, and truncated to the most recent events.

## Training and evaluation

### Pretraining

The decoder predicts the next code at each non-padding position. It uses AdamW, a one-cycle cosine learning-rate schedule, gradient clipping, validation perplexity, and early stopping on validation loss. The best decoder is saved to `pretrained_decoder.pt`.

### Fine-tuning

The classification head predicts all target conditions simultaneously. The decoder trains at `encoder_lr_scale × learning_rate`; the classification head uses the full learning rate. The default focal loss is weighted by the negative-to-positive ratio for each condition, clipped at `pos_weight_clip`.

Validation reports:

- Macro AUROC over labels with at least one positive example
- Mean average precision (mAP) over labels with at least one positive example
- Per-condition AUROC and average precision at the best checkpoint

Early stopping and checkpoint selection use validation mAP, which is generally more informative for sparse multi-label outcomes. Automatic mixed precision is enabled only for CUDA devices.

## Outputs

The output directory is created automatically and can contain:

| File | Description |
| --- | --- |
| `pretrained_decoder.pt` | Best next-event-pretrained decoder state dict |
| `best_model.pt` | Best complete classification model state dict by validation mAP |
| `predictions.csv` | Test patient IDs and one probability column per target code |
| `wandb/` | Local W&B run files when tracking is enabled |

Example prediction schema:

```csv
patient_id,44054006,195967001
patient-003,0.1247,0.0319
```

## Reproducibility and data safety

- The current pipeline does not set global random seeds. Model initialization, data-loader shuffling, and anchor sampling can therefore vary across runs.
- Augmented anchors use Python's `random` module. Add explicit Python, NumPy, and PyTorch seeds if exact reproducibility is required.
## License

No license file is currently included. Add a license before distributing or reusing the project outside its current context.
