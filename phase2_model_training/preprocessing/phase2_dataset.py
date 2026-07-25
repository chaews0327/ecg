"""
A2(grain-matched)용 Dataset.

data/ptbxl_dataset.py의 PTBXLDataset(Phase 1에서 만든 것)을 그대로 재사용하고,
그 위에 template beat / RR sequence 계산만 추가한다. raw ecg 로딩, 정규화,
라벨 파싱 등은 전부 기존 코드를 그대로 쓰고 새로 만들지 않는다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from data.ptbxl_dataset import PTBXLDataset, make_unified_dataset
from phase2_model_training.preprocessing.median_beat import extract_beat_variability_features, extract_template_beat
from phase2_model_training.preprocessing.rr_sequence import extract_rr_sequence


class GrainMatchedDataset(Dataset):
    """PTBXLDataset을 감싸서 template_beat / rr_seq / rr_mask를 추가로 반환."""

    def __init__(self, base_dataset: PTBXLDataset, fs: int = 500, rr_max_len: int = 32):
        self.base = base_dataset
        self.fs = fs
        self.rr_max_len = rr_max_len

        # 편의 속성 전달 (audit/train 스크립트가 label_col_groups 등에 접근할 수 있도록)
        self.label_col_groups = base_dataset.label_col_groups
        self.labels = base_dataset.labels
        self.label_names = base_dataset.label_names

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]  # ecg, ecg_id, patient_id, {task}_label 등은 기존 로직 그대로
        ecg_np = item["ecg"].numpy() if torch.is_tensor(item["ecg"]) else np.asarray(item["ecg"])

        template, n_beats = extract_template_beat(ecg_np, fs=self.fs)
        beat_stats = extract_beat_variability_features(ecg_np, fs=self.fs)
        rr_seq, rr_mask, n_valid = extract_rr_sequence(ecg_np, fs=self.fs, max_len=self.rr_max_len)

        item["template_beat"] = torch.from_numpy(template)
        item["beat_stats"] = torch.from_numpy(beat_stats)  # (3,): qrs_width_std, st_level_std, beat_shape_std
        item["rr_seq"] = torch.from_numpy(rr_seq)
        item["rr_mask"] = torch.from_numpy(rr_mask)
        item["n_beats_used"] = n_beats  # 모니터링용: fallback(0) 비율을 반드시 확인할 것

        return item


def make_grain_matched_dataset(split: str, ptbxl_root, split_dir,
                                fs: int = 500, rr_max_len: int = 32) -> GrainMatchedDataset:
    """Phase 1의 make_unified_dataset을 그대로 재사용해서 A2 dataset을 만든다."""
    base = make_unified_dataset(split=split, ptbxl_root=ptbxl_root, split_dir=split_dir, lead_swap=False)
    return GrainMatchedDataset(base, fs=fs, rr_max_len=rr_max_len)
