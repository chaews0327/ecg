"""
Sanity check: 실제 PTB-XL 데이터나 MERL/MELP checkpoint 없이도,
label_parser -> ptbxl_dataset -> linear_probe -> audit_basic -> audit_confound_control
로 이어지는 전체 파이프라인이 올바르게 배선되어 있는지 확인한다.

이 스크립트는:
  1) 축소된 가짜 PTB-XL(4개 ECG, 가짜 wfdb 신호)을 생성해 label_parser.py를 검증하고
  2) '의도적으로 leakage를 심은' 합성 embedding을 만들어 audit_confound_control.py가
     그 leakage를 실제로 탐지하는지 확인한다.

실제 연구를 시작하기 전에, 이 스크립트가 먼저 에러 없이 끝까지 도는지
확인하는 것을 권장한다:

    python -m tests.sanity_check
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SANITY_DIR = Path("tests/_sanity_tmp")


def _make_fake_ptbxl():
    """실제 PTB-XL과 동일한 스키마를 가진 초소형 가짜 데이터셋 생성."""
    import wfdb

    root = SANITY_DIR / "fake_ptbxl"
    (root / "records500").mkdir(parents=True, exist_ok=True)

    scp_rows = [
        ("NORM", 1, 0, 0, "NORM"),
        ("IMI", 1, 0, 0, "MI"),
        ("LVH", 1, 1, 0, "HYP"),  # diagnostic이면서 form이기도 한 실제 케이스 재현
        ("NDT", 0, 1, 0, ""),
        ("LVOLT", 0, 1, 0, ""),
        ("SR", 0, 0, 1, ""),
        ("AFIB", 0, 0, 1, ""),
        ("SBRAD", 0, 0, 1, ""),
    ]
    scp = pd.DataFrame(scp_rows, columns=["code", "diagnostic", "form", "rhythm", "diagnostic_class"])
    scp = scp.set_index("code")
    scp.to_csv(root / "scp_statements.csv")

    db_rows = [
        dict(ecg_id=1, patient_id=101, strat_fold=1, filename_lr="records100/1", filename_hr="records500/1",
             scp_codes="{'NORM': 100.0, 'SR': 0.0}"),
        dict(ecg_id=2, patient_id=102, strat_fold=9, filename_lr="records100/2", filename_hr="records500/2",
             scp_codes="{'LVH': 100.0, 'AFIB': 0.0, 'NDT': 50.0}"),
        dict(ecg_id=3, patient_id=103, strat_fold=10, filename_lr="records100/3", filename_hr="records500/3",
             scp_codes="{'IMI': 100.0, 'SBRAD': 0.0, 'LVOLT': 0.0}"),
        dict(ecg_id=4, patient_id=104, strat_fold=2, filename_lr="records100/4", filename_hr="records500/4",
             scp_codes="{'NORM': 100.0}"),
    ]
    pd.DataFrame(db_rows).to_csv(root / "ptbxl_database.csv", index=False)

    channel_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    rng = np.random.RandomState(0)
    for i in range(1, 5):
        sig = rng.randn(5000, 12).astype(np.float64) * 0.1
        wfdb.wrsamp(str(i), fs=500, sig_name=channel_names, p_signal=sig,
                    units=["mV"] * 12, fmt=["16"] * 12, write_dir=str(root / "records500"))

    return root


def check_label_parser_and_dataset():
    print("### [1/3] label_parser.py + ptbxl_dataset.py 배선 확인 ###")
    from data.label_parser import build_unified_labels, build_task_labels
    from data.ptbxl_dataset import make_unified_dataset, make_task_dataset

    ptbxl_root = _make_fake_ptbxl()
    split_dir = SANITY_DIR / "fake_split"

    # label_parser
    import json
    unified, code_lists = build_unified_labels(ptbxl_root)
    assert len(unified) == 4, f"unified row 수가 예상과 다름: {len(unified)}"
    assert unified.loc[unified.ecg_id == 2, "LVH"].iloc[0] == 1, "form+diagnostic 동시 케이스(LVH) 처리 오류"
    assert unified.loc[unified.ecg_id == 2, "HYP"].iloc[0] == 1, "LVH -> HYP superclass 매핑 오류"

    task_labels = build_task_labels(ptbxl_root)
    assert len(task_labels["form"][0]) == 2, "form 라벨이 없는 ECG가 필터링되지 않음"

    # csv로 저장 후 Dataset 로딩까지 확인
    from data.label_parser import save_splits
    save_splits(unified, split_dir / "unified", "unified")
    for task, (df, cols) in task_labels.items():
        save_splits(df, split_dir / task, task)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "label_columns.json", "w") as f:
        json.dump(code_lists, f)

    ds = make_unified_dataset("train", ptbxl_root=ptbxl_root, split_dir=split_dir, lead_swap=False)
    item = ds[0]
    assert item["ecg"].shape == (12, 5000), f"ecg shape 이상: {item['ecg'].shape}"
    assert item["form_label"].shape[0] == 3
    assert item["rhythm_label"].shape[0] == 3

    ds_task = make_task_dataset("rhythm", "train", ptbxl_root=ptbxl_root, split_dir=split_dir, lead_swap=True)
    assert ds_task.labels.shape[1] == 3

    print("  통과: label_parser / PTBXLDataset 정상 동작\n")


def check_linear_probe():
    print("### [2/3] linear_probe.py 자가 점검 ###")
    from eval.linear_probe import linear_probe_auc

    rng = np.random.RandomState(0)
    x = rng.randn(200, 8)
    y = np.concatenate([(x[:, 0:1] > 0).astype(float), (x[:, 1:2] > 0).astype(float)], axis=1)
    result = linear_probe_auc(x[:150], y[:150], x[150:], y[150:], label_names=["a", "b"])
    assert result["macro_auc"] > 0.9, "선형분리 가능한 데이터에서 AUC가 비정상적으로 낮음"
    print(f"  통과: macro_auc={result['macro_auc']:.3f} (선형분리 가능한 합성 데이터)\n")


def check_leakage_audit_detects_planted_leakage():
    print("### [3/3] audit_confound_control.py가 의도적으로 심은 leakage를 탐지하는지 확인 ###")
    import config as cfg
    from phase1_leakage_audit.audit_confound_control import run_confound_control

    rng = np.random.RandomState(0)

    def make_split(n):
        f_factor = rng.randint(0, 4, size=n)
        # form-rhythm 사이에 '약한' 진짜 상관관계만 부여 (완전 독립 아님, 완전 종속도 아님)
        r_factor = (f_factor + rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1])) % 3
        form = np.eye(4)[f_factor]
        rhythm = np.eye(3)[r_factor]
        # embedding에는 form/rhythm 정보를 '의도적으로 강하게' 섞어 넣음 (leakage 시뮬레이션)
        z_form_part = np.eye(4)[f_factor] * 3 + rng.randn(n, 4) * 0.3
        z_rhythm_part = np.eye(3)[r_factor] * 3 + rng.randn(n, 3) * 0.3
        z = np.concatenate([z_form_part, z_rhythm_part, rng.randn(n, 8) * 0.5], axis=1)
        super_ = np.eye(5)[rng.randint(0, 5, size=n)]
        return z, form, rhythm, super_

    for split, n in [("train", 300), ("test", 100)]:
        z, form, rhythm, super_ = make_split(n)
        np.savez(
            cfg.OUTPUT_DIR / f"_sanity_unified_{split}.npz",
            z=z, form_labels=form, rhythm_labels=rhythm, super_labels=super_,
            form_names=np.array(["F0", "F1", "F2", "F3"]),
            rhythm_names=np.array(["R0", "R1", "R2"]),
            super_names=np.array(["S0", "S1", "S2", "S3", "S4"]),
            ecg_id=np.arange(n),
        )

    result = run_confound_control("_sanity")
    assert result["leakage_score_rhythm"] > 0.1, (
        f"의도적으로 심은 leakage가 탐지되지 않음: leakage_score_rhythm={result['leakage_score_rhythm']:.4f}"
    )
    assert result["leakage_score_form"] > 0.1, (
        f"의도적으로 심은 leakage가 탐지되지 않음: leakage_score_form={result['leakage_score_form']:.4f}"
    )
    print("  통과: 의도적으로 심은 leakage가 정확히 탐지됨\n")


def main():
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        check_label_parser_and_dataset()
        check_linear_probe()
        check_leakage_audit_detects_planted_leakage()
        print("=" * 50)
        print("전체 파이프라인 배선 확인 완료. 실제 PTB-XL/checkpoint로 진행해도 좋습니다.")
        print("=" * 50)
    finally:
        shutil.rmtree(SANITY_DIR, ignore_errors=True)
        import config as cfg
        for split in ["train", "test"]:
            p = cfg.OUTPUT_DIR / f"_sanity_unified_{split}.npz"
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
