from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    data_dir: Path = Path("/workspace/transf/data")
    train_val_dir: Path = data_dir / "train_val"
    test_dir: Path = data_dir / "test"
    output_dir: Path = Path("output_transformer_decoder")

    # Data
    max_seq_len: int = 1024
    min_code_freq: int = 3

    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.15
    max_time_days: float = 3652.5
    num_types: int = 7
    n_time_features: int = 3
    classifier_dropout: float = 0.3
    classifier_hidden: int = 256
    num_genders: int = 3   # M, F, unknown
    num_races: int = 6     # white, black, asian, native, other, unknown

    # Finetuning
    batch_size: int = 32
    learning_rate: float = 1e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 0.01
    num_epochs: int = 60
    patience: int = 10
    warmup_ratio: float = 0.1
    pos_weight_clip: float = 50.0
    num_workers: int = 4

    # Loss
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    aux_lm_weight: float = 0.1

    # Logging
    use_wandb: bool = False

    # Pretraining
    do_pretrain: bool = True
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 64
    pretrain_patience: int = 10

    # Anchor augmentation
    do_augment: bool = True
    augment_n_per_patient: int = 2
    augment_earliest_offset_days: int = 365
    augment_latest_offset_days: int = 365
    augment_allzero_keep_frac: float = 0.0

    # Target codes
    target_codes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        df = pd.read_csv(self.data_dir / "target_conditions.csv")
        self.target_codes = df["CODE"].tolist()
