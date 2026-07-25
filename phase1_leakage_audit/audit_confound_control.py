"""
Phase 1 - Step 4 (가장 중요한 실험): Confound-controlled cross-leakage 측정.

문제의식:
    "z(embedding)가 rhythm 라벨을 잘 예측한다"는 결과만으로는, 그게
    (a) embedding이 부적절하게 rhythm 정보를 form-embedding에 섞어 쓴 것인지,
    (b) 원래 PTB-XL 데이터에서 form과 rhythm이 실제로 어느 정도 상관관계를
        갖기 때문에 생기는 정당한 결과인지 구분할 수 없다.

해법:
    "순수 라벨 상관관계만으로 설명 가능한 baseline"을 별도로 계산한다.
    이 baseline은 true_form_label -> true_rhythm_label 을 예측하는 성능이다
    (embedding을 전혀 쓰지 않고, 라벨-라벨 상관관계만 사용).

    leakage_score = AUC(z -> 반대 라벨) - AUC(진짜 라벨 -> 반대 라벨)

    leakage_score가 유의미하게 양수라면, embedding이 "원래 데이터에 존재하는
    상관관계를 넘어서는" 정보를 섞어 쓰고 있다는 정량적 증거가 된다.

실행:
    python -m phase1_leakage_audit.audit_confound_control
"""
import json

import numpy as np

import config
from eval.linear_probe import linear_probe_auc
from phase1_leakage_audit.audit_basic import load_npz


def run_confound_control(model_name: str) -> dict:
    train = load_npz(model_name, "train")
    test = load_npz(model_name, "test")

    # (1) embedding 기반 cross-task 예측
    cross_form_emb_to_rhythm = linear_probe_auc(
        train["z"], train["rhythm"], test["z"], test["rhythm"],
        label_names=train["rhythm_names"],
    )["macro_auc"]

    cross_rhythm_emb_to_form = linear_probe_auc(
        train["z"], train["form"], test["z"], test["form"],
        label_names=train["form_names"],
    )["macro_auc"]

    # (2) 순수 라벨 상관관계 baseline (embedding을 전혀 쓰지 않음)
    baseline_form_to_rhythm = linear_probe_auc(
        train["form"], train["rhythm"], test["form"], test["rhythm"],
        label_names=train["rhythm_names"],
    )["macro_auc"]

    baseline_rhythm_to_form = linear_probe_auc(
        train["rhythm"], train["form"], test["rhythm"], test["form"],
        label_names=train["form_names"],
    )["macro_auc"]

    leakage_rhythm = cross_form_emb_to_rhythm - baseline_form_to_rhythm
    leakage_form = cross_rhythm_emb_to_form - baseline_rhythm_to_form

    result = dict(
        cross_embedding_to_rhythm=cross_form_emb_to_rhythm,
        baseline_label_to_rhythm=baseline_form_to_rhythm,
        leakage_score_rhythm=leakage_rhythm,
        cross_embedding_to_form=cross_rhythm_emb_to_form,
        baseline_label_to_form=baseline_rhythm_to_form,
        leakage_score_form=leakage_form,
    )

    print(f"\n=== {model_name}: confound-controlled leakage ===")
    print(f"  z -> rhythm (embedding 사용)        : {cross_form_emb_to_rhythm:.4f}")
    print(f"  form_label -> rhythm (순수 상관관계)  : {baseline_form_to_rhythm:.4f}")
    print(f"  => leakage_score_rhythm             : {leakage_rhythm:+.4f}")
    print()
    print(f"  z -> form (embedding 사용)          : {cross_rhythm_emb_to_form:.4f}")
    print(f"  rhythm_label -> form (순수 상관관계)  : {baseline_rhythm_to_form:.4f}")
    print(f"  => leakage_score_form               : {leakage_form:+.4f}")

    return result


def main():
    all_results = {}
    for model_name in ["merl", "melp"]:
        path = config.OUTPUT_DIR / f"{model_name}_unified_train.npz"
        if not path.exists():
            print(f"[{model_name}] {path} 없음 -> extract_embeddings_{model_name}.py를 먼저 실행하세요. 건너뜁니다.")
            continue
        all_results[model_name] = run_confound_control(model_name)

    out_path = config.OUTPUT_DIR / "phase1_leakage_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n최종 요약 저장 -> {out_path}")

    print("\n판단 가이드:")
    print("  leakage_score >> 0  -> 문제가 실재함. Phase 2(grain-matched dual encoder)로 진행.")
    print("  leakage_score ~ 0   -> 이 모델/데이터에서는 뚜렷한 shortcut 증거가 약함.")
    print("                         다른 checkpoint/데이터셋에서도 재확인 후 다음 단계 강조점을 재조정할 것.")


if __name__ == "__main__":
    main()
