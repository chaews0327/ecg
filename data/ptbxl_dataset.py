"""
PTB-XL 12-lead ECG를 읽어오는 PyTorch Dataset.

전처리는 MERL(Liu et al., 2024)/MELP(Wang et al., 2025) 공개 코드의 방식을
그대로 따른다 (filename_hr=500Hz 레코드 사용, 12x5000 형태로 truncate,
전체 텐서에 대한 min-max 정규화). 이렇게 해야 두 checkpoint의 frozen encoder에
넣었을 때 각 논문이 보고한 성능과 어긋나지 않는다.

label_parser.py가 생성한 csv (unified 또는 task별 csv)를 그대로 읽는다.
어떤 컬럼이 라벨인지는 label_columns.json(또는 명시적으로 넘긴 label_col_groups)에서 가져온다.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        ptbxl_root: Path,
        label_col_groups: Dict[str, List[str]],
        lead_swap: bool = False,
        max_len: int = 5000,
    ):
        """
        Args:
            csv_path: label_parser.py가 만든 split csv 경로
                      (예: data_split/unified/unified_test.csv,
                           data_split/form/form_test.csv)
            ptbxl_root: PTB-XL 원본 데이터 루트 (records500/... 이 있는 폴더)
            label_col_groups: {"form": [...], "rhythm": [...], "super_class": [...]}
                               csv에 실제로 존재하는 그룹만 넣으면 됨
                               (예: form-only csv라면 {"form": [...]}만)
            lead_swap: True면 index 4,5(aVL/aVF)를 swap.
                       - MELP 공개 checkpoint용 전처리는 이 swap을 포함함
                         (MIMIC-IV-ECG 학습 시 lead 순서 차이 보정)
                       - MERL 공개 checkpoint용 전처리는 swap 없음
                       각 모델의 원 저장소 데이터셋 코드를 그대로 따른 것.
            max_len: 신호 길이 truncate 기준 (500Hz * 10s = 5000)
        """
        self.df = pd.read_csv(csv_path)
        self.ptbxl_root = Path(ptbxl_root)
        self.label_col_groups = label_col_groups
        self.lead_swap = lead_swap
        self.max_len = max_len

        # 각 그룹의 라벨 컬럼이 실제 csv에 있는지 확인
        for group, cols in label_col_groups.items():
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"csv에 '{group}' 라벨 컬럼이 없습니다: {missing}")

        # 편의 속성: 단일 그룹만 있는 경우(form-only, rhythm-only 등) .labels로 바로 접근
        if len(label_col_groups) == 1:
            only_group = next(iter(label_col_groups))
            self.labels = self.df[label_col_groups[only_group]].values.astype(np.float32)
            self.label_names = label_col_groups[only_group]
        else:
            self.labels = None
            self.label_names = None

    def __len__(self):
        return len(self.df)

    def _load_signal(self, filename_hr: str) -> np.ndarray:
        record_path = str(self.ptbxl_root / filename_hr)
        ecg = wfdb.rdsamp(record_path)[0]  # (T, 12)
        ecg = ecg.T  # (12, T)
        ecg = ecg[:, : self.max_len]
        # min-max normalize to [0, 1] (MERL/MELP 원 코드와 동일)
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg) + 1e-8)
        if self.lead_swap:
            ecg[[4, 5]] = ecg[[5, 4]]
        return ecg.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg = self._load_signal(row["filename_hr"])

        item = {
            "ecg": torch.from_numpy(ecg),
            "ecg_id": int(row["ecg_id"]),
            "patient_id": row["patient_id"],
        }

        for group, cols in self.label_col_groups.items():
            vec = row[cols].values.astype(np.float32)
            item[f"{group}_label"] = torch.from_numpy(vec)

        return item


def load_label_columns(split_dir: Path) -> Dict[str, List[str]]:
    """label_parser.py가 저장한 label_columns.json을 로드."""
    with open(Path(split_dir) / "label_columns.json") as f:
        return json.load(f)


def make_unified_dataset(split: str, ptbxl_root: Path, split_dir: Path,
                          lead_swap: bool = False) -> PTBXLDataset:
    """Phase 1(leakage audit)용: form+rhythm+super 라벨이 전부 붙은 dataset."""
    label_cols = load_label_columns(split_dir)
    csv_path = Path(split_dir) / "unified" / f"unified_{split}.csv"
    return PTBXLDataset(
        csv_path=csv_path,
        ptbxl_root=ptbxl_root,
        label_col_groups=label_cols,  # {"form": [...], "rhythm": [...], "super_class": [...]}
        lead_swap=lead_swap,
    )


def make_task_dataset(task: str, split: str, ptbxl_root: Path, split_dir: Path,
                       lead_swap: bool = False) -> PTBXLDataset:
    """Phase 2(모델 학습/평가)용: 특정 task(form/rhythm/super_class)만 필터링된 dataset."""
    label_cols = load_label_columns(split_dir)
    csv_path = Path(split_dir) / task / f"{task}_{split}.csv"
    return PTBXLDataset(
        csv_path=csv_path,
        ptbxl_root=ptbxl_root,
        label_col_groups={task: label_cols[task]},
        lead_swap=lead_swap,
    )
