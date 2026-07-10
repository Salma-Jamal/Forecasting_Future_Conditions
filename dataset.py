from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(
        self,
        patient_ids: List[str],
        sequences: Dict[str, Dict],
    ):
        self.patient_ids = patient_ids
        self.sequences = sequences

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        seq = self.sequences[pid]

        return {
            "input_ids": torch.tensor(seq["input_ids"], dtype=torch.long),
            "type_ids": torch.tensor(seq["type_ids"], dtype=torch.long),
            "time_features": torch.tensor(seq["time_features"], dtype=torch.float),
            "attention_mask": torch.tensor(seq["attention_mask"], dtype=torch.long),
            "gender_id": torch.tensor(seq.get("gender_id", 2), dtype=torch.long),
            "race_id": torch.tensor(seq.get("race_id", 5), dtype=torch.long),
            "patient_id": pid,
        }


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        patient_ids: List[str],
        sequences: Dict[str, Dict],
        labels_df,
        target_codes: List[str],
    ):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.target_codes = target_codes

        self.label_map = {}
        if labels_df is not None and not labels_df.empty:
            for _, row in labels_df.iterrows():
                pid = row["patient_id"]
                vec = [float(row[code]) for code in target_codes]
                self.label_map[pid] = vec

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        seq = self.sequences[pid]

        input_ids = torch.tensor(seq["input_ids"], dtype=torch.long)
        type_ids = torch.tensor(seq["type_ids"], dtype=torch.long)
        time_feat = torch.tensor(seq["time_features"], dtype=torch.float)
        attn_mask = torch.tensor(seq["attention_mask"], dtype=torch.long)

        labels = self.label_map.get(pid, [0.0] * len(self.target_codes))
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "type_ids": type_ids,
            "time_features": time_feat,
            "attention_mask": attn_mask,
            "gender_id": torch.tensor(seq.get("gender_id", 2), dtype=torch.long),
            "race_id": torch.tensor(seq.get("race_id", 5), dtype=torch.long),
            "labels": labels,
            "patient_id": pid,
        }
