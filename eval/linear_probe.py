"""
Frozen feature -> label 의 linear probe 평가.

이 함수는 "embedding -> label" 뿐 아니라 "label -> label"(예: true form label ->
rhythm label 예측) 에도 그대로 재사용된다. 첫 번째 인자가 embedding이든
label matrix든 그냥 (N, D) 실수 행렬로 취급하기 때문이다.
이 재사용성이 Phase 1의 confound-controlled leakage 측정에서 핵심적으로 쓰인다.

멀티라벨(한 샘플이 여러 클래스를 동시에 가질 수 있음)을 가정하고,
클래스마다 독립적인 이진 로지스틱 회귀를 학습해 macro-averaged AUC를 계산한다.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def linear_probe_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
    max_iter: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Args:
        x_train, x_test: (N, D) 실수 feature (embedding일 수도, label matrix일 수도 있음)
        y_train, y_test: (N, C) {0,1} multi-hot label
        label_names: 길이 C인 클래스 이름 리스트 (없으면 class_0, class_1, ...)

    Returns:
        {
          "per_class_auc": {label_name: auc, ...},   # train/test 양쪽에 두 클래스가
                                                       # 모두 존재하는 라벨만 포함
          "macro_auc": float,                          # per_class_auc의 평균
          "skipped_labels": [...]                      # 클래스 불균형이 심해 AUC 계산이
                                                          # 불가능해 건너뛴 라벨
        }
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]

    n_classes = y_train.shape[1]
    if label_names is None:
        label_names = [f"class_{i}" for i in range(n_classes)]
    assert len(label_names) == n_classes, "label_names 길이가 y의 클래스 수와 다릅니다"

    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    per_class_auc = {}
    skipped = []

    for c in range(n_classes):
        name = label_names[c]
        y_tr_c = y_train[:, c]
        y_te_c = y_test[:, c]

        # train이나 test에 한 클래스만 있으면 AUC 계산이 불가능하므로 건너뜀
        if len(np.unique(y_tr_c)) < 2 or len(np.unique(y_te_c)) < 2:
            skipped.append(name)
            continue

        clf = LogisticRegression(
            max_iter=max_iter, class_weight="balanced", random_state=seed
        )
        clf.fit(x_train_s, y_tr_c)
        prob = clf.predict_proba(x_test_s)[:, 1]
        per_class_auc[name] = float(roc_auc_score(y_te_c, prob))

    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "skipped_labels": skipped,
    }


def incremental_auc(
    covariate_train: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    covariate_test: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
    max_iter: int = 2000,
    seed: int = 42,
) -> Dict:
    """Nested-model incremental leakage test.

    단순히 "z -> y" 성능과 "covariate -> y" 성능을 따로 학습해서 빼는 것은
    불공정한 비교다(z가 covariate보다 정보량이 원래 훨씬 많은 고차원
    embedding이기 때문에, 그냥 "더 좋은 표현이라 잘 맞추는 것"과 "진짜
    covariate를 넘어서는 shortcut"을 구분하지 못한다).

    대신 covariate를 이미 통제 변수로 넣은 두 nested 모델을 비교한다:
        Model 1 (baseline) : y ~ covariate
        Model 2 (augmented): y ~ [covariate ; z]   (concat)

    Model 2가 Model 1보다 유의미하게 나으면, 그건 covariate가 이미 설명하는
    부분을 넘어서서 z가 추가 정보를 갖고 있다는 뜻이다 -- 이게 "진짜 leakage"에
    훨씬 가까운 정의다.

    Returns:
        {
          "baseline_auc": Model 1의 macro AUC,
          "augmented_auc": Model 2의 macro AUC,
          "incremental_auc": augmented_auc - baseline_auc,
          "per_class": {label: {"baseline":..., "augmented":..., "incremental":...}}
        }
    """
    baseline_result = linear_probe_auc(
        covariate_train, y_train, covariate_test, y_test,
        label_names=label_names, max_iter=max_iter, seed=seed,
    )

    x_train_aug = np.concatenate([covariate_train, z_train], axis=1)
    x_test_aug = np.concatenate([covariate_test, z_test], axis=1)
    augmented_result = linear_probe_auc(
        x_train_aug, y_train, x_test_aug, y_test,
        label_names=label_names, max_iter=max_iter, seed=seed,
    )

    per_class = {}
    for name in baseline_result["per_class_auc"]:
        if name in augmented_result["per_class_auc"]:
            b = baseline_result["per_class_auc"][name]
            a = augmented_result["per_class_auc"][name]
            per_class[name] = {"baseline": b, "augmented": a, "incremental": a - b}

    return {
        "baseline_auc": baseline_result["macro_auc"],
        "augmented_auc": augmented_result["macro_auc"],
        "incremental_auc": augmented_result["macro_auc"] - baseline_result["macro_auc"],
        "per_class": per_class,
    }


if __name__ == "__main__":
    # 빠른 자가 점검: 완전히 분리 가능한 합성 데이터로 macro_auc가 1.0 근처인지 확인
    rng = np.random.RandomState(0)
    n = 200
    x = rng.randn(n, 8)
    y = (x[:, 0:1] > 0).astype(float)  # 첫 feature가 라벨을 완전히 결정
    y = np.concatenate([y, (x[:, 1:2] > 0).astype(float)], axis=1)

    x_tr, x_te = x[:150], x[150:]
    y_tr, y_te = y[:150], y[150:]

    result = linear_probe_auc(x_tr, y_tr, x_te, y_te, label_names=["easy_0", "easy_1"])
    print(result)
    assert result["macro_auc"] > 0.9, "self-check 실패: 선형분리 가능한 데이터에서 AUC가 낮게 나옴"
    print("self-check 통과")
