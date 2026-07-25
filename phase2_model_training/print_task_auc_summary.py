"""
run_phase2.sh의 [3/5] 단계에서 쓰는 A0/A1/A2 task 성능 요약 출력 스크립트.
bash -c 인라인 문자열 안에 복잡한 f-string/이스케이프 따옴표를 섞으면
쉘 파싱 오류가 나기 쉬워서, 별도 파일로 분리했다.
"""
import numpy as np

import config
from eval.linear_probe import linear_probe_auc


def main():
    for model in ["a0", "a1", "a2"]:
        train = dict(np.load(str(config.OUTPUT_DIR / f"phase2_{model}_train.npz"), allow_pickle=True))
        test = dict(np.load(str(config.OUTPUT_DIR / f"phase2_{model}_test.npz"), allow_pickle=True))

        key = "z" if model == "a0" else "z_form"
        res_form = linear_probe_auc(
            train[key], train["form_labels"], test[key], test["form_labels"],
            label_names=list(train["form_names"]),
        )

        key_r = "z" if model == "a0" else "z_rhythm"
        res_rhythm = linear_probe_auc(
            train[key_r], train["rhythm_labels"], test[key_r], test["rhythm_labels"],
            label_names=list(train["rhythm_names"]),
        )

        print(f"[{model}] form macro AUC = {res_form['macro_auc']:.4f}   "
              f"rhythm macro AUC = {res_rhythm['macro_auc']:.4f}")


if __name__ == "__main__":
    main()