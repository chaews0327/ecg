"""
Phase 2 sanity check: 전처리(median beat, RR sequence), label similarity,
incremental leakage 로직을 실제 PTB-XL 없이 합성 데이터로 검증한다.

주의: soft_supcon_loss(torch 구현)과 A0/A1/A2 모델 자체의 forward/backward는
이 환경에 torch가 없어 여기서 직접 실행 검증하지 못했다. 대신 loss의 수식은
순수 numpy로 별도 프로토타입해서 "라벨이 비슷한 쌍을 가깝게 배치하면 loss가
낮아진다"는 것을 확인했고, soft_supcon.py는 그 수식을 그대로 옮긴 것이다.
아래를 반드시 먼저 실행해서 실제 환경에서 torch 배선이 맞는지 확인할 것:

    python -m phase2_model_training.train_phase2 --model a0 --sanity_check
    python -m phase2_model_training.train_phase2 --model a1 --sanity_check
    python -m phase2_model_training.train_phase2 --model a2 --sanity_check

실행:
    python -m tests.sanity_check_phase2
"""
import numpy as np

import config
from phase2_model_training.preprocessing.median_beat import detect_r_peaks, extract_template_beat
from phase2_model_training.preprocessing.rr_sequence import extract_rr_sequence, rr_summary_stats
from phase2_model_training.losses.label_similarity import label_overlap_similarity, combine_similarity
from eval.linear_probe import incremental_auc
from phase2_model_training.audit_incremental_leakage import run_incremental_audit


def _make_synthetic_ecg(fs=500, duration_s=10, hr=75, seed=0):
    T = fs * duration_s
    rr_interval_s = 60.0 / hr
    true_r_times = np.arange(0.3, duration_s, rr_interval_s)
    true_r_samples = (true_r_times * fs).astype(int)
    true_r_samples = true_r_samples[true_r_samples < T]

    rng = np.random.RandomState(seed)
    ecg = np.zeros((12, T))
    for lead in range(12):
        amp = 0.5 + lead * 0.1
        for r in true_r_samples:
            width = int(0.02 * fs)
            idx = np.arange(max(0, r - 3 * width), min(T, r + 3 * width))
            ecg[lead, idx] += amp * np.exp(-0.5 * ((idx - r) / width) ** 2)
        ecg[lead] += rng.randn(T) * 0.02
    return ecg, true_r_samples, fs, hr


def check_preprocessing():
    print("### [1/4] median_beat.py / rr_sequence.py ###")
    ecg, true_r_samples, fs, hr = _make_synthetic_ecg()

    r_peaks = detect_r_peaks(ecg[1], fs=fs)
    assert abs(len(r_peaks) - len(true_r_samples)) <= 1, "R-peak 개수가 크게 다름"

    template, n_beats = extract_template_beat(ecg, fs=fs)
    assert template.shape == (12, 300)
    assert n_beats >= len(true_r_samples) - 2

    rr_seq, rr_mask, n_valid = extract_rr_sequence(ecg, fs=fs)
    stats = rr_summary_stats(rr_seq, rr_mask)
    assert abs(stats["hr_bpm"] - hr) < 3, f"추정 HR({stats['hr_bpm']:.1f})이 실제({hr})와 크게 다름"

    print(f"  통과: R-peak {len(r_peaks)}개 검출, template shape={template.shape}, "
          f"추정 HR={stats['hr_bpm']:.1f} (실제 {hr})\n")


def check_label_similarity():
    print("### [2/4] label_similarity.py (overlap + co-occurrence semantic) ###")
    from phase2_model_training.losses.label_similarity import (
        build_cooccurrence_similarity_matrix,
        label_to_pairwise_via_lookup,
    )

    y = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    s = label_overlap_similarity(y)
    assert abs(s[0, 1] - 0.5) < 1e-8
    assert s[0, 2] == 0.0
    assert np.allclose(s, combine_similarity(s))

    y_all = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]])
    class_sim = build_cooccurrence_similarity_matrix(y_all)
    assert class_sim[0, 1] > 0.5 and class_sim[0, 2] == 0.0
    sem = label_to_pairwise_via_lookup(y, class_sim)
    combined = combine_similarity(s, sem, alpha=0.7, beta=0.3)
    assert combined.shape == (3, 3)
    print("  통과: Jaccard overlap + co-occurrence semantic similarity 계산 정상\n")


def check_incremental_auc():
    print("### [3/4] incremental_auc (핵심 방법론 수정) ###")
    rng = np.random.RandomState(0)
    n = 400
    f = rng.randint(0, 3, size=n)
    r = (f + rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])) % 3
    cov = np.eye(3)[f]
    y = np.eye(3)[r]

    z_no_leak = np.eye(3)[f] * 2 + rng.randn(n, 3) * 1.5
    z_leak = np.eye(3)[f] * 2 + np.eye(3)[r] * 3 + rng.randn(n, 3) * 0.5

    n_tr = 300
    res_no_leak = incremental_auc(cov[:n_tr], z_no_leak[:n_tr], y[:n_tr],
                                   cov[n_tr:], z_no_leak[n_tr:], y[n_tr:])
    res_leak = incremental_auc(cov[:n_tr], z_leak[:n_tr], y[:n_tr],
                                cov[n_tr:], z_leak[n_tr:], y[n_tr:])

    assert res_no_leak["incremental_auc"] < 0.05, "leakage 없는 시나리오인데 incremental_auc가 큼"
    assert res_leak["incremental_auc"] > 0.15, "leakage 있는 시나리오인데 incremental_auc가 작음"
    print(f"  통과: no-leak incremental_auc={res_no_leak['incremental_auc']:.3f}, "
          f"leak incremental_auc={res_leak['incremental_auc']:.3f} -> 두 시나리오가 명확히 구분됨\n")


def check_audit_incremental_leakage_pipeline():
    print("### [4/4] audit_incremental_leakage.py 배선(A1 vs A2 비교) ###")
    rng = np.random.RandomState(0)
    n_train, n_test = 300, 100
    n = n_train + n_test

    form_factor = rng.randint(0, 4, size=n)
    rhythm_factor = (form_factor + rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.1, 0.1])) % 3
    form = np.eye(4)[form_factor]
    rhythm = np.eye(3)[rhythm_factor]

    z_form_a1 = np.eye(4)[form_factor] * 2 + np.eye(3, 4)[rhythm_factor % 3] * 2.5 + rng.randn(n, 4) * 0.5
    z_form_a2 = np.eye(4)[form_factor] * 2 + rng.randn(n, 4) * 0.5

    def split(z):
        names = dict(form_names=["F0", "F1", "F2", "F3"], rhythm_names=["R0", "R1", "R2"])
        train = dict(form=form[:n_train], rhythm=rhythm[:n_train], z_form=z[:n_train], **names)
        test = dict(form=form[n_train:], rhythm=rhythm[n_train:], z_form=z[n_train:], **names)
        return train, test

    tr1, te1 = split(z_form_a1)
    tr2, te2 = split(z_form_a2)

    res_a1 = run_incremental_audit(tr1, te1, "z_form")
    res_a2 = run_incremental_audit(tr2, te2, "z_form")

    assert res_a1["incremental_leakage_into_rhythm"] > res_a2["incremental_leakage_into_rhythm"]
    print(f"  통과: grain-mismatch 시뮬레이션(A1) leakage={res_a1['incremental_leakage_into_rhythm']:.3f} "
          f"> grain-matched 시뮬레이션(A2) leakage={res_a2['incremental_leakage_into_rhythm']:.3f}\n")


def main():
    check_preprocessing()
    check_label_similarity()
    check_incremental_auc()
    check_audit_incremental_leakage_pipeline()
    print("=" * 60)
    print("Phase 2 데이터/방법론 로직 검증 완료.")
    print("다음으로 반드시 실행: python -m phase2_model_training.train_phase2 --model a0 --sanity_check")
    print("(torch 모델 forward/backward는 이 환경에 torch가 없어 위 명령으로 직접 확인 필요)")
    print("=" * 60)


if __name__ == "__main__":
    main()
