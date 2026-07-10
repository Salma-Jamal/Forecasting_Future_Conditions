from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _wandb_log(data: dict, step: Optional[int] = None):
    """Log to wandb if a run is active; no-op otherwise."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)


def macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores) if scores else 0.0


def mean_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            scores.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores) if scores else 0.0


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (auroc_list, ap_list) per label column; np.nan for classes with no positives."""
    n = y_true.shape[1]
    aurocs = np.full(n, np.nan)
    aps = np.full(n, np.nan)
    for i in range(n):
        if y_true[:, i].sum() > 0:
            aurocs[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
            aps[i] = average_precision_score(y_true[:, i], y_pred[:, i])
    return aurocs, aps


def compute_pos_weight(labels_df, target_codes, clip: float = 50.0) -> torch.Tensor:
    if labels_df is None or labels_df.empty:
        return None
    pos = []
    for code in target_codes:
        p = max(float((labels_df[code] == 1).sum()), 1.0)
        n = max(float((labels_df[code] == 0).sum()), 1.0)
        pos.append(min(n / p, clip))
    return torch.tensor(pos, dtype=torch.float)


class FocalLoss(nn.Module):
    """Multi-label focal loss with per-class alpha weighting.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified examples, focusing gradient on hard ones.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # numerically stable BCE in log space (no sigmoid → no overflow)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # p_t = sigmoid*logit if target=1 else 1-sigmoid
        modulating = (1.0 - p_t) ** self.gamma
        loss = modulating * bce

        if self.alpha.dim() == 0:
            loss = self.alpha * loss
        else:
            loss = loss * self.alpha.unsqueeze(0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Pretrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.use_wandb = getattr(cfg, "use_wandb", False)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.pretrain_lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = None
        self.use_amp = (device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Pretrain", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            type_ids = batch["type_ids"].to(self.device)
            time_feat = batch["time_features"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            gender_id = batch.get("gender_id")
            if gender_id is not None:
                gender_id = gender_id.to(self.device)
            race_id = batch.get("race_id")
            if race_id is not None:
                race_id = race_id.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(input_ids, type_ids, time_feat, attn_mask,
                                        gender_id=gender_id, race_id=race_id)
                    loss = self.model.compute_pretrain_loss(logits, input_ids)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, type_ids, time_feat, attn_mask,
                                    gender_id=gender_id, race_id=race_id)
                loss = self.model.compute_pretrain_loss(logits, input_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Val", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            type_ids = batch["type_ids"].to(self.device)
            time_feat = batch["time_features"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            gender_id = batch.get("gender_id")
            if gender_id is not None:
                gender_id = gender_id.to(self.device)
            race_id = batch.get("race_id")
            if race_id is not None:
                race_id = race_id.to(self.device)
            logits = self.model(input_ids, type_ids, time_feat, attn_mask,
                                gender_id=gender_id, race_id=race_id)
            loss = self.model.compute_pretrain_loss(logits, input_ids)
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        perplexity = np.exp(min(avg_loss, 20.0))
        return {"loss": avg_loss, "perplexity": perplexity}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        total_steps = num_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.pretrain_lr,
            total_steps=total_steps,
            pct_start=self.cfg.warmup_ratio,
            anneal_strategy="cos",
        )
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"PPL: {val_metrics['perplexity']:.2f}"
            )
            _wandb_log({
                "pretrain/epoch": epoch + 1,
                "pretrain/train_loss": train_loss,
                "pretrain/val_loss": val_metrics["loss"],
                "pretrain/perplexity": val_metrics["perplexity"],
            })
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "pretrained_decoder.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.pretrain_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        self.model.load_state_dict(torch.load(self.output_dir / "pretrained_decoder.pt"))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: torch.device,
        output_dir: Path,
        pos_weight: torch.Tensor = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        # Separate parameter groups: lower LR for pretrained encoder, full LR for head.
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": cfg.learning_rate * cfg.encoder_lr_scale},
                {"params": head_params, "lr": cfg.learning_rate},
            ],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = None

        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        if getattr(cfg, "use_focal_loss", False):
            alpha = pos_weight if pos_weight is not None else torch.tensor(1.0)
            self.criterion = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma)
            print(f"Using FocalLoss (gamma={cfg.focal_gamma})")
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.aux_lm_weight = getattr(cfg, "aux_lm_weight", 0.0)
        self.use_amp = (device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        self.best_metric = 0.0
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.use_wandb = getattr(cfg, "use_wandb", False)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Train", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            type_ids = batch["type_ids"].to(self.device)
            time_feat = batch["time_features"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            gender_id = batch.get("gender_id")
            if gender_id is not None:
                gender_id = gender_id.to(self.device)
            race_id = batch.get("race_id")
            if race_id is not None:
                race_id = race_id.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, lm_logits = self.model(
                        input_ids, type_ids, time_feat, attn_mask,
                        gender_id=gender_id, race_id=race_id,
                        return_lm_logits=True,
                    )
                    task_loss = self.criterion(logits, labels)
                    if self.aux_lm_weight > 0:
                        lm_loss = self.model.decoder.compute_pretrain_loss(
                            lm_logits, input_ids
                        )
                        loss = task_loss + self.aux_lm_weight * lm_loss
                    else:
                        loss = task_loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, lm_logits = self.model(
                    input_ids, type_ids, time_feat, attn_mask,
                    gender_id=gender_id, race_id=race_id,
                    return_lm_logits=True,
                )
                task_loss = self.criterion(logits, labels)
                if self.aux_lm_weight > 0:
                    lm_loss = self.model.decoder.compute_pretrain_loss(
                        lm_logits, input_ids
                    )
                    loss = task_loss + self.aux_lm_weight * lm_loss
                else:
                    loss = task_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_labels = []
        all_probs = []

        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            type_ids = batch["type_ids"].to(self.device)
            time_feat = batch["time_features"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            gender_id = batch.get("gender_id")
            if gender_id is not None:
                gender_id = gender_id.to(self.device)
            race_id = batch.get("race_id")
            if race_id is not None:
                race_id = race_id.to(self.device)

            logits = self.model(input_ids, type_ids, time_feat, attn_mask,
                                gender_id=gender_id, race_id=race_id)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        auroc = macro_auroc(all_labels, all_probs)
        ap = mean_ap(all_labels, all_probs)
        per_auroc, per_ap = per_class_metrics(all_labels, all_probs)

        return {"macro_auroc": auroc, "mAP": ap,
                "per_auroc": per_auroc, "per_ap": per_ap}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int,
            target_codes=None):
        total_steps = num_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in self.optimizer.param_groups],
            total_steps=total_steps,
            pct_start=self.cfg.warmup_ratio,
            anneal_strategy="cos",
        )
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val AUROC: {val_metrics['macro_auroc']:.4f} | "
                f"Val mAP: {val_metrics['mAP']:.4f}"
            )

            _wandb_log({
                "finetune/epoch": epoch + 1,
                "finetune/train_loss": train_loss,
                "finetune/val_macro_auroc": val_metrics["macro_auroc"],
                "finetune/val_mAP": val_metrics["mAP"],
            })

            # Checkpoint + early-stop on mAP (more informative than AUROC for
            # imbalanced multi-label incident-condition prediction).
            if val_metrics["mAP"] > self.best_metric:
                self.best_metric = val_metrics["mAP"]
                self.best_auroc = val_metrics["macro_auroc"]
                self.patience_counter = 0
                self._best_per_ap = val_metrics["per_ap"]
                self._best_per_auroc = val_metrics["per_auroc"]
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(
            torch.load(self.output_dir / "best_model.pt")
        )

        # Per-class breakdown at the best (saved) checkpoint.
        per_ap = getattr(self, "_best_per_ap", None)
        per_auroc = getattr(self, "_best_per_auroc", None)
        if per_ap is not None and target_codes is not None:
            print(f"\n--- Per-class metrics at best mAP={self.best_metric:.4f} ---")
            order = np.argsort(-per_ap)  # highest AP first
            print(f"{'code':>14s} {'AP':>7s} {'AUROC':>7s}")
            per_class_data = []
            for i in order:
                if np.isnan(per_ap[i]):
                    continue
                print(f"{str(target_codes[i]):>14s} {per_ap[i]:7.3f} {per_auroc[i]:7.3f}")
                per_class_data.append({
                    "code": str(target_codes[i]),
                    "AP": float(per_ap[i]),
                    "AUROC": float(per_auroc[i]),
                })
            n_eval = int((~np.isnan(per_ap)).sum())
            print(f"({n_eval}/{len(target_codes)} classes with >=1 positive in val)")

            _wandb_log({
                "finetune/best_val_mAP": self.best_metric,
                "finetune/best_val_auroc": self.best_auroc,
            })
            if per_class_data:
                _wandb_log({
                    "per_class_metrics": wandb.Table(
                        data=[[d["code"], d["AP"], d["AUROC"]] for d in per_class_data],
                        columns=["code", "AP", "AUROC"],
                    )
                })

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_probs = []
        for batch in tqdm(loader, desc="Predict", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            type_ids = batch["type_ids"].to(self.device)
            time_feat = batch["time_features"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            gender_id = batch.get("gender_id")
            if gender_id is not None:
                gender_id = gender_id.to(self.device)
            race_id = batch.get("race_id")
            if race_id is not None:
                race_id = race_id.to(self.device)

            logits = self.model(input_ids, type_ids, time_feat, attn_mask,
                                gender_id=gender_id, race_id=race_id)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)
