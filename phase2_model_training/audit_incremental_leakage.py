"""
Incremental(nested-model) leakage test를 모든 모델에 일관되게 적용해서 비교한다.

기존 audit_confound_control.py(단순 subtraction)의 한계:
    "z -> rhythm AUC" 와 "true_form_label -> rhythm AUC"를 따로 학습해서 빼는 것은
    z가 form_label보다 원래 정보량이 훨씬 많은 고차원 embedding이라는 걸 무시한
    불공정한 비교였다. 이러면 "z가 그냥 더 좋은 표현이라 잘 맞추는 것"과
    "z가 진짜 부적절한 shortcut을 쓰는 것"이 구분되지 않는다.

이 스크립트는 eval/linear_probe.py의 incremental_auc()로 이를 교체한다:
    Model 1: rhythm ~ true_form_label
    Model 2: rhythm ~ [true_form_label ; z]
    incremental_auc = AUC(Model 2) - AUC(Model 1)

그리고 중요한 재해석 포인트:
    - MERL/MELP의 z는 애초에 "form만 담겠다"고 주장한 적 없는 monolithic
      embedding이므로, 여기서 나오는 incremental_auc는 "결함"이 아니라
      "monolithic 표현의 자연스러운 특성"에 가깝다 (fair reference point로만 사용).
    - 반면 A1/A2의 z_form은 "form만 담아야 한다"고 설계 의도가 명시된
      표현이므로, 여기서 incremental_auc가 높게 나오면 그건 설계 의도를
      위반하는 진짜 문제다. A0/A1/A2는 동일한 데이터/학습 파이프라인으로
      학습되었으므로 이 비교가 훨씬 공정하다.

실행:
    python -m phase2_model_training.audit_incremental_leakage
"""
import json

import numpy as np

import config
from eval.linear_probe import incremental_auc


def _load_phase1_npz(model_name: str, split: str):
    path = config.OUTPUT_DIR / f"{model_name}_unified_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "z": d["z"], "form": d["form_labels"], "rhythm": d["rhythm_labels"],
        "form_names": list(d["form_names"]), "rhythm_names": list(d["rhythm_names"]),
    }


def _load_phase2_npz(model_name: str, split: str):
    path = config.OUTPUT_DIR / f"phase2_{model_name}_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    result = {
        "form": d["form_labels"], "rhythm": d["rhythm_labels"],
        "form_names": list(d["form_names"]), "rhythm_names": list(d["rhythm_names"]),
    }
    if "z" in d:
        result["z"] = d["z"]
    if "z_form" in d:
        result["z_form"] = d["z_form"]
    if "z_rhythm" in d:
        result["z_rhythm"] = d["z_rhythm"]
    return result


def run_incremental_audit(train: dict, test: dict, z_key: str) -> dict:
    """z_key로 가리키는 embedding에 대해 양방향(form->rhythm 넘어서는 leakage,
    rhythm->form 넘어서는 leakage) incremental_auc를 계산."""
    res_rhythm = incremental_auc(
        covariate_train=train["form"], z_train=train[z_key], y_train=train["rhythm"],
        covariate_test=test["form"], z_test=test[z_key], y_test=test["rhythm"],
        label_names=train["rhythm_names"],
    )
    res_form = incremental_auc(
        covariate_train=train["rhythm"], z_train=train[z_key], y_train=train["form"],
        covariate_test=test["rhythm"], z_test=test[z_key], y_test=test["form"],
        label_names=train["form_names"],
    )
    return {
        "incremental_leakage_into_rhythm": res_rhythm["incremental_auc"],
        "incremental_leakage_into_form": res_form["incremental_auc"],
        "baseline_auc_rhythm": res_rhythm["baseline_auc"],
        "augmented_auc_rhythm": res_rhythm["augmented_auc"],
        "baseline_auc_form": res_form["baseline_auc"],
        "augmented_auc_form": res_form["augmented_auc"],
    }


def main():
    all_results = {}

    # --- 참고용: MERL/MELP monolithic z (Phase 1) ---
    for model_name in ["merl", "melp"]:
        train = _load_phase1_npz(model_name, "train")
        test = _load_phase1_npz(model_name, "test")
        if train is None or test is None:
            print(f"[{model_name}] Phase 1 결과 없음, 건너뜀")
            continue
        result = run_incremental_audit(train, test, "z")
        all_results[f"{model_name}_monolithic_z (참고용, 설계상 form-only가 아님)"] = result

    # --- 핵심: 우리가 학습한 A0/A1/A2 (동일 파이프라인이라 공정 비교 가능) ---
    # A0는 z 하나뿐이라 그대로. A1/A2는 z_form과 z_rhythm을 각각 따로 감사해야
    # "rhythm branch 자체가 얼마나 깨끗한가"를 볼 수 있음 (이전엔 z_form만 돌려서
    # z_rhythm 쪽 결과가 아예 존재하지 않았음).
    audit_targets = [
        ("a0", "z"),
        ("a1", "z_form"),
        ("a1", "z_rhythm"),   # <- 추가
        ("a2", "z_form"),
        ("a2", "z_rhythm"),   # <- 추가
    ]

    for model_name, key in audit_targets:
        train = _load_phase2_npz(model_name, "train")
        test = _load_phase2_npz(model_name, "test")
        if train is None or test is None:
            print(f"[phase2_{model_name}] embedding 없음 -> extract_embeddings_phase2.py를 먼저 실행하세요. 건너뜀")
            continue
        if key not in train or key not in test:
            print(f"[phase2_{model_name}] '{key}' 키가 npz에 없음 (a0는 z_form/z_rhythm이 없는 게 정상), 건너뜀")
            continue

        result = run_incremental_audit(train, test, key)

        if model_name == "a0":
            desc = "(모든 정보를 담아도 되는 monolithic 표현)"
        elif key == "z_form":
            desc = "(설계상 form-only여야 함 -- 핵심 지표는 incremental_leakage_into_rhythm)"
        else:  # key == "z_rhythm"
            desc = "(설계상 rhythm-only여야 함 -- 핵심 지표는 incremental_leakage_into_form)"

        label = f"phase2_{model_name}_{key} {desc}"
        all_results[label] = result

    print("\n" + "=" * 70)
    print("Incremental Leakage 비교 (양수가 클수록, covariate로 설명 안 되는")
    print("추가 정보를 embedding이 담고 있다는 뜻)")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"\n[{name}]")
        print(f"  incremental_leakage_into_rhythm = {res['incremental_leakage_into_rhythm']:+.4f}  "
              f"(baseline={res['baseline_auc_rhythm']:.4f} -> augmented={res['augmented_auc_rhythm']:.4f})")
        print(f"  incremental_leakage_into_form   = {res['incremental_leakage_into_form']:+.4f}  "
              f"(baseline={res['baseline_auc_form']:.4f} -> augmented={res['augmented_auc_form']:.4f})")

    out_path = config.OUTPUT_DIR / "phase2_incremental_leakage_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n저장 -> {out_path}")

    print("\n판단 가이드:")
    print("  z_form에서는 incremental_leakage_into_rhythm이 낮을수록 좋음(rhythm이 안 새는 것).")
    print("  z_rhythm에서는 incremental_leakage_into_form이 낮을수록 좋음(form이 안 새는 것).")
    print("  (반대 방향, 즉 z_form의 into_form이나 z_rhythm의 into_rhythm이 높은 건 당연하고")
    print("   오히려 정상입니다 -- 그건 leakage가 아니라 해당 branch가 제 역할을 하고 있다는 뜻.)")
    print("  phase2_a1_z_form 대비 phase2_a2_z_form, 그리고 phase2_a1_z_rhythm 대비")
    print("  phase2_a2_z_rhythm의 leakage가 각각 뚜렷이 낮다면, grain matching이 양쪽")
    print("  축 모두에서 실제로 더 깨끗하게 분리한다는 정량적 증거.")


if __name__ == "__main__":
    main()