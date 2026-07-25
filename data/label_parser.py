"""
PTB-XL의 scp_statements.csv / ptbxl_database.csv로부터
Form(19) / Rhythm(12) / Diagnostic Superclass(5) multi-hot 라벨을 만든다.

PTB-XL은 71개 SCP-ECG statement를 diagnostic(44) / form(19) / rhythm(12)
세 카테고리로 분류한다 (Wagner et al., Scientific Data 2020).
이 세 카테고리는 scp_statements.csv의 'diagnostic'/'form'/'rhythm' 불리언 컬럼으로
표시되어 있으므로, 여기서는 그 컬럼을 그대로 읽어서 라벨을 구성한다.

두 가지 산출물을 만든다:
  1. build_task_labels()   -> task(form/rhythm/super_class)별로 "해당 task에
     라벨이 하나라도 있는 ECG만" 남긴 DataFrame. MERL/MELP 등 기존 문헌과
     동일한 방식이라, Phase 2(모델 학습)에서 비교 가능성을 위해 사용.
  2. build_unified_labels() -> 필터링 없이 모든 ECG에 대해 form/rhythm/super
     라벨을 전부 붙인 단일 DataFrame. Phase 1의 cross-leakage audit은
     "같은 ECG에 대한 form 라벨과 rhythm 라벨"이 동시에 필요하므로 이쪽을 쓴다.

사용법 (CLI):
    python -m data.label_parser --ptbxl_root /path/to/ptbxl --out_dir ./data_split
"""
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import config


def _load_raw(ptbxl_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ptbxl_database.csv, scp_statements.csv를 읽고 scp_codes를 dict로 파싱."""
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    return db, scp


def _get_code_lists(scp: pd.DataFrame) -> Dict[str, List[str]]:
    """scp_statements.csv에서 form/rhythm/diagnostic 카테고리별 statement code 목록 추출."""
    form_codes = scp[scp["form"] == 1].index.tolist()
    rhythm_codes = scp[scp["rhythm"] == 1].index.tolist()
    diag_codes = scp[scp["diagnostic"] == 1].index.tolist()
    return {"form": form_codes, "rhythm": rhythm_codes, "diagnostic": diag_codes}


def _multi_hot(scp_dict: dict, code_list: List[str]) -> List[int]:
    present = set(scp_dict.keys())
    return [1 if c in present else 0 for c in code_list]


def _diagnostic_superclass_vector(scp_dict: dict, diag_scp: pd.DataFrame,
                                   superclasses: List[str]) -> List[int]:
    """PTB-XL 공식 aggregate_diagnostic 로직의 multi-hot 버전.

    diag_scp: scp_statements.csv에서 diagnostic==1인 행만 (index=code,
              column 'diagnostic_class'를 가짐)
    """
    classes = set()
    for code in scp_dict.keys():
        if code in diag_scp.index:
            cls = diag_scp.loc[code, "diagnostic_class"]
            if isinstance(cls, str) and cls in superclasses:
                classes.add(cls)
    return [1 if c in classes else 0 for c in superclasses]


def build_unified_labels(ptbxl_root: Path = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """필터링 없이 모든 ECG에 form/rhythm/super 라벨을 전부 붙인 단일 테이블 생성.

    Returns:
        df: ecg_id, patient_id, strat_fold, filename_lr, filename_hr,
            form 컬럼(19개), rhythm 컬럼(12개), super 컬럼(5개)
        code_lists: {"form": [...], "rhythm": [...], "super_class": [...]}
    """
    ptbxl_root = Path(ptbxl_root or config.PTBXL_ROOT)
    db, scp = _load_raw(ptbxl_root)
    codes = _get_code_lists(scp)
    form_codes, rhythm_codes = codes["form"], codes["rhythm"]
    diag_scp = scp[scp["diagnostic"] == 1]
    superclasses = config.SUPERCLASSES

    rows = []
    for ecg_id, row in db.iterrows():
        scp_dict = row["scp_codes"]
        f_vec = _multi_hot(scp_dict, form_codes)
        r_vec = _multi_hot(scp_dict, rhythm_codes)
        s_vec = _diagnostic_superclass_vector(scp_dict, diag_scp, superclasses)

        record = dict(
            ecg_id=ecg_id,
            patient_id=row["patient_id"],
            strat_fold=int(row["strat_fold"]),
            filename_lr=row["filename_lr"],
            filename_hr=row["filename_hr"],
        )
        record.update(dict(zip(form_codes, f_vec)))
        record.update(dict(zip(rhythm_codes, r_vec)))
        record.update(dict(zip(superclasses, s_vec)))
        rows.append(record)

    df = pd.DataFrame(rows)
    code_lists = {"form": form_codes, "rhythm": rhythm_codes, "super_class": superclasses}
    return df, code_lists


def build_task_labels(ptbxl_root: Path = None) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    """task(form/rhythm/super_class)별로 라벨이 하나라도 있는 ECG만 남긴 테이블.

    MERL(Liu et al., 2024)이 공개한 data_split/ptbxl/{form,rhythm,super_class}/*.csv
    와 동일한 필터링 규칙(해당 task label이 all-zero인 행 제거)을 따른다.
    """
    unified, code_lists = build_unified_labels(ptbxl_root)
    result = {}
    base_cols = ["ecg_id", "patient_id", "strat_fold", "filename_lr", "filename_hr"]

    for task, cols in code_lists.items():
        sub = unified[base_cols + cols].copy()
        has_label = sub[cols].sum(axis=1) > 0
        sub = sub[has_label].reset_index(drop=True)
        result[task] = (sub, cols)

    return result


def split_by_fold(df: pd.DataFrame):
    """PTB-XL 공식 protocol: fold 1-8 train / fold 9 val / fold 10 test."""
    train = df[df["strat_fold"].isin(config.TRAIN_FOLDS)].reset_index(drop=True)
    val = df[df["strat_fold"] == config.VAL_FOLD].reset_index(drop=True)
    test = df[df["strat_fold"] == config.TEST_FOLD].reset_index(drop=True)
    return train, val, test


def save_splits(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = split_by_fold(df)
    train.to_csv(out_dir / f"{name}_train.csv", index=False)
    val.to_csv(out_dir / f"{name}_val.csv", index=False)
    test.to_csv(out_dir / f"{name}_test.csv", index=False)
    print(f"[{name}] train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, default=str(config.PTBXL_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(config.SPLIT_DIR))
    args = parser.parse_args()

    ptbxl_root = Path(args.ptbxl_root)
    out_dir = Path(args.out_dir)

    # (1) task별 필터링된 split (Phase 2에서 사용, 기존 문헌과 비교 가능)
    task_labels = build_task_labels(ptbxl_root)
    for task, (df, cols) in task_labels.items():
        save_splits(df, out_dir / task, task)

    # (2) unified split (Phase 1 leakage audit용)
    unified, code_lists = build_unified_labels(ptbxl_root)
    save_splits(unified, out_dir / "unified", "unified")

    # 라벨 컬럼 목록을 메타데이터로 저장 (하드코딩 없이 재사용하기 위함)
    with open(out_dir / "label_columns.json", "w") as f:
        json.dump(code_lists, f, indent=2, ensure_ascii=False)
    print(f"label_columns.json 저장 완료 -> {out_dir / 'label_columns.json'}")


if __name__ == "__main__":
    main()
