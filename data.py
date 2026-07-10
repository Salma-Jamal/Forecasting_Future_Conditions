import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config


def _parse_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    return df


# (table_name, date_col, type_id) for every event source we tokenize.
EVENT_SOURCES = [
    ("conditions", "START", 0),
    ("medications", "START", 1),
    ("procedures", "DATE", 2),
    ("observations", "DATE", 3),
    ("encounters", "START", 4),
    ("careplans", "START", 5),
    ("immunizations", "DATE", 6),
]


class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.splits: pd.DataFrame = None
        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.test_ids: List[str] = []

        self.train_val: Dict[str, pd.DataFrame] = {}
        self.test: Dict[str, pd.DataFrame] = {}

        self.train_val_anchors: Dict[str, pd.Timestamp] = {}
        self.test_anchors: Dict[str, pd.Timestamp] = {}

        self.birthdates: Dict[str, pd.Timestamp] = {}
        self.genders: Dict[str, int] = {}
        self.races: Dict[str, int] = {}

    GENDER_MAP = {"M": 0, "F": 1}
    RACE_MAP = {"white": 0, "black": 1, "asian": 2, "native": 3, "other": 4}

    def load(self):
        self._load_splits()
        self._load_tables("train_val")
        self._load_tables("test")
        self._compute_train_val_anchors()
        self._load_test_anchors()
        self._load_demographics()

    def _load_splits(self):
        self.splits = pd.read_csv(self.cfg.data_dir / "patient_splits.csv")
        self.train_ids = self.splits.query("split == 'train'")["Id"].tolist()
        self.val_ids = self.splits.query("split == 'val'")["Id"].tolist()
        self.test_ids = self.splits.query("split == 'test'")["Id"].tolist()

    TABLE_NAMES = [
        "patients", "encounters", "conditions", "observations",
        "medications", "procedures", "allergies", "immunizations",
        "careplans", "devices", "imaging_studies",
    ]

    DATE_COLUMNS = {
        "patients": ["BIRTHDATE", "DEATHDATE"],
        "encounters": ["START", "STOP"],
        "conditions": ["START", "STOP"],
        "observations": ["DATE"],
        "medications": ["START", "STOP"],
        "procedures": ["DATE"],
        "allergies": ["START", "STOP"],
        "immunizations": ["DATE"],
        "careplans": ["START", "STOP"],
        "devices": ["START", "STOP"],
        "imaging_studies": ["DATE"],
    }

    def _load_tables(self, split: str):
        base = self.cfg.train_val_dir if split == "train_val" else self.cfg.test_dir
        tables = self.train_val if split == "train_val" else self.test
        for name in self.TABLE_NAMES:
            path = base / f"{name}.csv"
            if path.exists():
                df = pd.read_csv(path, low_memory=False)
                date_cols = self.DATE_COLUMNS.get(name, [])
                df = _parse_dates(df, date_cols)
                df.columns = [c.upper() for c in df.columns]
                tables[name] = df

    def _compute_train_val_anchors(self):
        enc = self.train_val.get("encounters")
        if enc is None or enc.empty:
            return
        all_ids = self.train_ids + self.val_ids
        sub = enc[enc["PATIENT"].isin(all_ids)].copy()
        sub["effective_end"] = sub["STOP"].fillna(sub["START"])
        last = sub.groupby("PATIENT")["effective_end"].max()
        anchors = last - pd.DateOffset(years=5)
        self.train_val_anchors = anchors.to_dict()

    def _load_test_anchors(self):
        path = self.cfg.data_dir / "test_anchors.csv"
        if path.exists():
            df = pd.read_csv(path)
            anchors = pd.to_datetime(df["anchor_date"])
            if anchors.dt.tz is not None:
                anchors = anchors.dt.tz_localize(None)
            self.test_anchors = dict(zip(df["Id"], anchors))
        else:
            enc = self.test.get("encounters")
            if enc is not None and not enc.empty:
                sub = enc[enc["PATIENT"].isin(self.test_ids)].copy()
                sub["effective_end"] = sub["STOP"].fillna(sub["START"])
                last = sub.groupby("PATIENT")["effective_end"].max()
                anchors = last - pd.DateOffset(years=5)
                self.test_anchors = anchors.to_dict()

    def _load_demographics(self):
        for split_tables in (self.train_val, self.test):
            pts = split_tables.get("patients")
            if pts is None or pts.empty:
                continue
            for _, row in pts.iterrows():
                pid = row["ID"]
                bd = row.get("BIRTHDATE")
                if bd is not None and not pd.isna(bd):
                    self.birthdates[pid] = bd
                g = str(row.get("GENDER", "")).strip()
                self.genders[pid] = self.GENDER_MAP.get(g, 2)  # 2 = unknown
                r = str(row.get("RACE", "")).strip().lower()
                self.races[pid] = self.RACE_MAP.get(r, 5)      # 5 = unknown


class LabelBuilder:
    def __init__(self, cfg: Config, loader: DataLoader):
        self.cfg = cfg
        self.loader = loader
        self.target_set = set(cfg.target_codes)

    def build(self, patient_ids: List[str], source: str) -> pd.DataFrame:
        tables = self.loader.train_val if source == "train_val" else self.loader.test
        anchors_dict = (
            self.loader.train_val_anchors if source == "train_val"
            else self.loader.test_anchors
        )

        cond = tables.get("conditions")
        if cond is None:
            return pd.DataFrame({"patient_id": patient_ids, **{
                code: 0 for code in self.cfg.target_codes
            }})

        rows = []
        for pid in patient_ids:
            rows.append({"patient_id": pid})
        result = pd.DataFrame(rows)
        for code in self.cfg.target_codes:
            result[code] = 0

        sub = cond[(cond["PATIENT"].isin(patient_ids)) &
                    (cond["CODE"].isin(self.target_set))].copy()
        if sub.empty:
            return result

        for pid in patient_ids:
            anchor = anchors_dict.get(pid)
            if anchor is None:
                continue
            pre_existing = set(
                sub[(sub["PATIENT"] == pid) & (sub["START"] < anchor)]["CODE"]
            )
            window_end = anchor + pd.DateOffset(years=5)
            incident = set(
                sub[(sub["PATIENT"] == pid) &
                    (sub["START"] >= anchor) &
                    (sub["START"] < window_end) &
                    (~sub["CODE"].isin(pre_existing))]["CODE"]
            )
            for code in incident:
                result.loc[result["patient_id"] == pid, code] = 1

        return result


class SequenceTokenizer:
    SPECIAL = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vocab: Dict[str, int] = dict(self.SPECIAL)
        self.id_to_code: Dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
        self.code_descriptions: Dict[str, str] = {}

    def fit(self, loader: DataLoader):
        counter = Counter()
        desc_map = {}
        for table_name, _, _ in EVENT_SOURCES:
            df = loader.train_val.get(table_name)
            if df is None or df.empty:
                continue
            sub = df[df["PATIENT"].isin(loader.train_ids)]
            if sub.empty or "CODE" not in sub.columns:
                continue
            counter.update(sub["CODE"])
            if "DESCRIPTION" in sub.columns:
                for _, row in sub.iterrows():
                    code = row["CODE"]
                    if code not in desc_map:
                        desc_map[code] = str(row["DESCRIPTION"])
        for code, count in counter.most_common():
            if count >= self.cfg.min_code_freq:
                self.vocab[code] = len(self.vocab)
                self.id_to_code[self.vocab[code]] = code
                if code in desc_map:
                    self.code_descriptions[code] = desc_map[code]
            if len(self.vocab) >= 10000:
                break

    def encode_code(self, code: str) -> int:
        return self.vocab.get(code, self.SPECIAL["[MASK]"])

    def __len__(self):
        return len(self.vocab)


def _pad_seq(tokens, types, times, mask, max_len, gender_id=2, race_id=5):
    n = len(tokens)
    if n == 0:
        times = [[0.0, 0.0, 0.0]]
        tokens = [0]
        types = [0]
        mask = [1]
        n = 1
    pad = max_len - n
    pad_times = [[0.0, 0.0, 0.0] for _ in range(pad)]
    return {
        "input_ids": (tokens + [0] * pad)[:max_len],
        "type_ids": (types + [0] * pad)[:max_len],
        "time_features": (times + pad_times)[:max_len],
        "attention_mask": (mask + [0] * pad)[:max_len],
        "gender_id": gender_id,
        "race_id": race_id,
    }


def _build_time_features(events, max_time_days, birthdate):
    """events: list of (tid, type_id, days_before_anchor, date)."""
    feats = []
    prev_date = None
    for e in events:
        date = e[3]
        days_before = e[2]
        f0 = min(max(days_before, 0) / max_time_days, 1.0)
        if birthdate is not None and not pd.isna(date) and not pd.isna(birthdate):
            age_days = (date - birthdate).days
            f1 = min(max(age_days, 0) / 36525.0, 1.0)
        else:
            f1 = 0.0
        if prev_date is not None and not pd.isna(date) and not pd.isna(prev_date):
            elapsed = (date - prev_date).days
            f2 = min(max(elapsed, 0) / 365.0, 1.0)
        else:
            f2 = 0.0
        feats.append([f0, f1, f2])
        prev_date = date
    return feats


def _collect_patient_events(pid, tables, tokenizer, anchor, max_time_days):
    """Collect events before anchor from all event sources."""
    events = []
    for table_name, date_col, etype in EVENT_SOURCES:
        table = tables.get(table_name)
        if table is None or table.empty:
            continue
        sub = table[table["PATIENT"] == pid]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            d = row.get(date_col)
            if pd.isna(d):
                continue
            days = (anchor - d).days
            if not np.isnan(days) and days >= 0:
                tid = tokenizer.encode_code(row["CODE"])
                events.append((tid, etype, int(days), d))
    events.sort(key=lambda x: x[3])
    return events


def _events_to_seq(events, max_seq_len, max_time_days, birthdate):
    # keep the most recent events (closest to anchor)
    events = events[-max_seq_len:]
    input_ids = [e[0] for e in events]
    type_ids = [e[1] for e in events]
    time_feat = _build_time_features(events, max_time_days, birthdate)
    attention_mask = [1] * len(input_ids)
    return input_ids, type_ids, time_feat, attention_mask


def generate_anchor_augmentation(
    patient_ids: List[str],
    loader: DataLoader,
    tokenizer: SequenceTokenizer,
    max_seq_len: int,
    max_time_days: float,
    target_set: set,
    n_per_patient: int = 8,
    skip_labels: bool = False,
    earliest_offset_days: int = 365,
    latest_offset_days: int = 365,
) -> Tuple[List[str], Dict[str, Dict], pd.DataFrame]:
    tables = loader.train_val
    cond = tables.get("conditions")

    augmented_pids = []
    augmented_sequences = {}
    augmented_label_rows = []

    for pid in patient_ids:
        real_anchor = loader.train_val_anchors.get(pid)
        if real_anchor is None:
            continue

        events_cache = _collect_patient_events(
            pid, tables, tokenizer, real_anchor, max_time_days
        )
        if not events_cache:
            continue

        earliest_date = events_cache[0][3]
        birthdate = loader.birthdates.get(pid)
        gender_id = loader.genders.get(pid, 2)
        race_id = loader.races.get(pid, 5)

        latest_anchor = real_anchor - pd.DateOffset(days=latest_offset_days)
        earliest_anchor = earliest_date + pd.DateOffset(days=earliest_offset_days)

        total_days = (latest_anchor - earliest_anchor).days
        if total_days < 0:
            continue

        n = min(n_per_patient, total_days + 1)
        sampled_offsets = sorted(random.sample(range(total_days + 1), n))

        pid_cond = cond[cond["PATIENT"] == pid].copy() if cond is not None and not skip_labels else None

        for aug_idx, day_offset in enumerate(sampled_offsets):
            cur_anchor = earliest_anchor + pd.DateOffset(days=day_offset)
            aug_pid = f"{pid}_aug_{aug_idx}"

            pre = [e for e in events_cache if e[3] < cur_anchor]
            if not pre:
                continue

            input_ids, type_ids, time_feat, attn_mask = _events_to_seq(
                pre, max_seq_len, max_time_days, birthdate
            )
            augmented_sequences[aug_pid] = _pad_seq(
                input_ids, type_ids, time_feat, attn_mask, max_seq_len,
                gender_id, race_id,
            )

            if not skip_labels:
                window_end = cur_anchor + pd.DateOffset(days=1825)
                if pid_cond is not None:
                    pre_existing = set(
                        pid_cond[(pid_cond["START"] < cur_anchor) &
                                 (pid_cond["CODE"].isin(target_set))]["CODE"]
                    )
                    incident = set(
                        pid_cond[(pid_cond["START"] >= cur_anchor) &
                                 (pid_cond["START"] < window_end) &
                                 (pid_cond["CODE"].isin(target_set)) &
                                 (~pid_cond["CODE"].isin(pre_existing))]["CODE"]
                    )
                else:
                    incident = set()

                label_row = {"patient_id": aug_pid}
                for code in target_set:
                    label_row[code] = 1 if code in incident else 0
                augmented_label_rows.append(label_row)

            augmented_pids.append(aug_pid)

    aug_labels = pd.DataFrame(augmented_label_rows) if augmented_label_rows else pd.DataFrame()
    return augmented_pids, augmented_sequences, aug_labels


def build_patient_sequences(
    patient_ids: List[str],
    loader: DataLoader,
    tokenizer: SequenceTokenizer,
    source: str,
    max_seq_len: int,
    max_time_days: float,
) -> Dict[str, Dict]:
    tables = loader.train_val if source == "train_val" else loader.test
    anchors = (
        loader.train_val_anchors if source == "train_val"
        else loader.test_anchors
    )

    result = {}
    for pid in patient_ids:
        anchor = anchors.get(pid)
        if anchor is None:
            result[pid] = _pad_seq([0], [0], [[0.0, 0.0, 0.0]], [1], max_seq_len,
                                   loader.genders.get(pid, 2), loader.races.get(pid, 5))
            continue

        events = _collect_patient_events(pid, tables, tokenizer, anchor, max_time_days)
        birthdate = loader.birthdates.get(pid)
        gender_id = loader.genders.get(pid, 2)
        race_id = loader.races.get(pid, 5)

        input_ids, type_ids, time_feat, attention_mask = _events_to_seq(
            events, max_seq_len, max_time_days, birthdate
        )

        result[pid] = _pad_seq(
            input_ids, type_ids, time_feat, attention_mask, max_seq_len,
            gender_id, race_id,
        )

    return result
