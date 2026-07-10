import argparse
import sys
from pathlib import Path
from dataclasses import asdict

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from data import (
    DataLoader as EHRLoader,
    LabelBuilder,
    SequenceTokenizer,
    build_patient_sequences,
    generate_anchor_augmentation,
)
from dataset import PretrainDataset, PatientSequenceDataset
from model import EHRDecoder, EHRDecoderForClassification
from trainer import Pretrainer, Trainer, compute_pos_weight

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/workspace/transf/data")
    parser.add_argument("--output-dir", type=str, default="/workspace/transf/output_transformer_decoder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="Path to pretrained decoder checkpoint. Skips pretrain if provided.")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pretraining phase entirely.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging.")
    parser.add_argument("--wandb-project", type=str, default="ehr-transformer",
                        help="wandb project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="wandb run name (auto-generated if not set).")
    args = parser.parse_args()

    cfg = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        use_wandb=args.wandb,
    )

    # ---------- Init wandb ----------
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("Warning: --wandb requested but wandb not installed. Skipping.")
        else:
            run_name = args.wandb_run_name or (
                f"seq{cfg.max_seq_len}_bs{cfg.batch_size}_"
                f"ep{cfg.num_epochs}_gamma{cfg.focal_gamma}"
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=asdict(cfg),
                dir=str(cfg.output_dir),
            )
            print(f"wandb run: {wandb.run.name} ({wandb.run.url})")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Target conditions: {len(cfg.target_codes)}")

    # ---------- Load data ----------
    loader = EHRLoader(cfg)
    loader.load()
    print(
        f"Train: {len(loader.train_ids)}, "
        f"Val: {len(loader.val_ids)}, "
        f"Test: {len(loader.test_ids)}"
    )

    tokenizer = SequenceTokenizer(cfg)
    tokenizer.fit(loader)
    print(f"Vocabulary size: {len(tokenizer)}")

    print("Building patient sequences...")
    train_seq = build_patient_sequences(
        loader.train_ids, loader, tokenizer, "train_val",
        cfg.max_seq_len, cfg.max_time_days,
    )
    val_seq = build_patient_sequences(
        loader.val_ids, loader, tokenizer, "train_val",
        cfg.max_seq_len, cfg.max_time_days,
    )
    test_seq = build_patient_sequences(
        loader.test_ids, loader, tokenizer, "test",
        cfg.max_seq_len, cfg.max_time_days,
    )

    # ---------- Phase 1: Pretrain (if enabled) ----------
    if not args.skip_pretrain:
        print("\n=== Phase 1: Pretraining (next-event prediction) ===")

        train_val_seq_pt = build_patient_sequences(
            loader.train_ids + loader.val_ids, loader, tokenizer, "train_val",
            cfg.max_seq_len, cfg.max_time_days,
        )

        train_pt_ids = loader.train_ids + loader.val_ids
        train_pt_seq = train_val_seq_pt

        pretrain_ds = PretrainDataset(train_pt_ids, train_pt_seq)
        val_pretrain_ids = loader.val_ids
        pretrain_val_ds = PretrainDataset(val_pretrain_ids, train_val_seq_pt)

        print(f"  Pretrain train size: {len(pretrain_ds)}")
        print(f"  Pretrain val size:   {len(pretrain_val_ds)}")

        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=cfg.pretrain_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        )
        pretrain_val_loader = DataLoader(
            pretrain_val_ds, batch_size=cfg.pretrain_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        )

        decoder = EHRDecoder(
            vocab_size=len(tokenizer),
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
            num_types=cfg.num_types,
            n_time_features=cfg.n_time_features,
            num_genders=cfg.num_genders,
            num_races=cfg.num_races,
        )
        total_p = sum(p.numel() for p in decoder.parameters())
        print(f"Decoder params: {total_p:,}")

        pt = Pretrainer(decoder, cfg, device, cfg.output_dir)
        pt.fit(pretrain_loader, pretrain_val_loader, cfg.pretrain_epochs)
        print(f"Best val loss: {pt.best_loss:.4f}")

        pretrain_ckpt = cfg.output_dir / "pretrained_decoder.pt"
    else:
        pretrain_ckpt = None

    # ---------- Phase 2: Finetune (classification) ----------
    print("\n=== Phase 2: Finetuning ===")

    decoder = EHRDecoder(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        num_types=cfg.num_types,
        n_time_features=cfg.n_time_features,
        num_genders=cfg.num_genders,
        num_races=cfg.num_races,
    )

    ckpt_path = args.pretrain_ckpt or pretrain_ckpt
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading pretrained decoder from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = decoder.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (freshly init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
    else:
        print("No pretrained checkpoint found. Training from scratch.")

    model = EHRDecoderForClassification(
        decoder, len(cfg.target_codes),
        classifier_dropout=cfg.classifier_dropout,
        classifier_hidden=cfg.classifier_hidden,
    )
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_p:,} (trainable: {train_p:,})")

    label_builder = LabelBuilder(cfg, loader)
    train_labels = label_builder.build(loader.train_ids, "train_val")
    val_labels = label_builder.build(loader.val_ids, "train_val")
    test_labels = label_builder.build(loader.test_ids, "test")

    # Anchor augmentation: slide anchor backward to multiply training examples
    aug_pids, aug_seqs, aug_labels = [], {}, None
    if cfg.do_augment:
        print("Generating multi-anchor augmented training examples...")
        aug_pids, aug_seqs, aug_labels = generate_anchor_augmentation(
            loader.train_ids,
            loader,
            tokenizer,
            cfg.max_seq_len,
            cfg.max_time_days,
            set(cfg.target_codes),
            n_per_patient=cfg.augment_n_per_patient,
            earliest_offset_days=cfg.augment_earliest_offset_days,
            latest_offset_days=cfg.augment_latest_offset_days,
        )
        print(f"Augmented dataset: {len(aug_pids)} generated examples")

        # Filter augmented examples: keep all with >=1 positive label, plus a
        # small fraction of all-zero examples (so the model still sees some
        # "healthy" contexts rather than only positive-augmented examples).
        if aug_labels is not None and not aug_labels.empty:
            has_pos = aug_labels[cfg.target_codes].sum(axis=1) > 0
            pos_pids = aug_labels.loc[has_pos, "patient_id"].tolist()
            zero_rows = aug_labels.loc[~has_pos]
            n_keep_zero = int(len(zero_rows) * cfg.augment_allzero_keep_frac)
            if n_keep_zero > 0:
                zero_keep = zero_rows.sample(n=n_keep_zero, random_state=42)
                zero_pids = zero_keep["patient_id"].tolist()
            else:
                zero_pids = []
            keep_pids = set(pos_pids + zero_pids)
            aug_pids = [p for p in aug_pids if p in keep_pids]
            aug_seqs = {p: aug_seqs[p] for p in aug_pids}
            aug_labels = aug_labels[aug_labels["patient_id"].isin(keep_pids)] \
                .reset_index(drop=True)
            print(f"  After filtering: {len(aug_pids)} kept "
                  f"({len(pos_pids)} with labels + {n_keep_zero} all-zero kept "
                  f"of {len(zero_rows)} all-zero generated)")

    merged_train_ids = loader.train_ids + aug_pids
    merged_train_seq = {**train_seq, **aug_seqs}
    if aug_labels is not None and not aug_labels.empty:
        merged_train_labels = pd.concat(
            [train_labels, aug_labels], ignore_index=True
        )
    else:
        merged_train_labels = train_labels

    train_ds = PatientSequenceDataset(
        merged_train_ids, merged_train_seq, merged_train_labels, cfg.target_codes
    )
    val_ds = PatientSequenceDataset(
        loader.val_ids, val_seq, val_labels, cfg.target_codes
    )
    test_ds = PatientSequenceDataset(
        loader.test_ids, test_seq, test_labels, cfg.target_codes
    )

    print(f"  Finetune train size: {len(train_ds)} ({len(loader.train_ids)} real + {len(aug_pids)} augmented)")
    print(f"  Finetune val size:   {len(val_ds)}")
    print(f"  Finetune test size:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    pos_weight = compute_pos_weight(merged_train_labels, cfg.target_codes, cfg.pos_weight_clip)
    trainer = Trainer(model, cfg, device, cfg.output_dir, pos_weight=pos_weight)
    trainer.fit(train_loader, val_loader, cfg.num_epochs, target_codes=cfg.target_codes)
    print(f"\nBest val macro AUROC: {trainer.best_auroc:.4f}")
    print(f"Best val mAP:         {trainer.best_metric:.4f}")

    # ---------- Predict & save ----------
    # NOTE: Test labels are withheld by the task setup (test anchors are after
    # the last available data, so LabelBuilder yields all-zero labels). We only
    # generate predictions for external scoring; do NOT report local test
    # metrics as they would always be 0.0.
    test_probs = trainer.predict(test_loader)
    pred_df = pd.DataFrame(test_probs, columns=cfg.target_codes)
    pred_df.insert(0, "patient_id", loader.test_ids)
    pred_df.to_csv(cfg.output_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved to {cfg.output_dir / 'predictions.csv'}")

    if args.wandb and _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
