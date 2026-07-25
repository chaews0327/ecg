"""
Phase 1 - Step 3: 기본 leakage probe.

extract_embeddings_{merl,melp}.py가 만든 outputs/{model}_unified_{split}.npz를
읽어서, embedding이 form 라벨과 rhythm 라벨을 각각 얼마나 잘 예측하는지
(self-task 성능)를 확인한다. 아직 cross-leakage는 아니다 -- 그건
audit_confound_control.py에서 다룬다.

실행:
    python -m phase1_leakage_audit.audit_basic
"""
import json

import numpy as np

import config
from eval.linear_probe import linear_probe_auc


def load_npz(model_name: str, split: str):
    path = config.OUTPUT_DIR / f"{model_name}_unified_{split}.npz"
    d = np.load(path, allow_pickle=True)
    return {
        "z": d["z"],
        "form": d["form_labels"],
        "rhythm": d["rhythm_labels"],
        "super": d["super_labels"],
        "form_names": list(d["form_names"]),
        "rhythm_names": list(d["rhythm_names"]),
        "super_names": list(d["super_names"]),
    }


def run_basic_audit(model_name: str) -> dict:
    train = load_npz(model_name, "train")
    test = load_npz(model_name, "test")

    results = {}
    for task, name_key in [("form", "form_names"), ("rhythm", "rhythm_names"), ("super", "super_names")]:
        res = linear_probe_auc(
            train["z"], train[task], test["z"], test[task],
            label_names=train[name_key],
        )
        results[task] = res
        print(f"[{model_name}] {task:6s} macro AUC = {res['macro_auc']:.4f}  "
              f"(skipped {len(res['skipped_labels'])} labels: {res['skipped_labels']})")

    return results


def main():
    all_results = {}
    for model_name in ["merl", "melp"]:
        path = config.OUTPUT_DIR / f"{model_name}_unified_train.npz"
        if not path.exists():
            print(f"[{model_name}] {path} 없음 -> extract_embeddings_{model_name}.py를 먼저 실행하세요. 건너뜁니다.")
            continue
        print(f"\n=== {model_name} ===")
        all_results[model_name] = run_basic_audit(model_name)

    out_path = config.OUTPUT_DIR / "phase1_basic_audit.json"
    # macro_auc, skipped_labels 등 요약만 저장 (per_class_auc는 audit_confound_control 쪽에서 상세 확인)
    summary = {
        m: {t: {"macro_auc": r["macro_auc"], "skipped_labels": r["skipped_labels"]}
            for t, r in tasks.items()}
        for m, tasks in all_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장 -> {out_path}")


if __name__ == "__main__":
    main()
