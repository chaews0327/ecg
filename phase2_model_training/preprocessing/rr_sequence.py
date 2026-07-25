"""
RR interval sequence 추출.

Rhythm branch의 입력은 "beat 하나"가 아니라 "beat 간의 관계"여야 한다는
설계 원칙에 따라, median_beat.py와 동일한 R-peak 검출 결과를 재사용해서
beat-to-beat 시계열(RR interval sequence)을 만든다.
"""
from typing import Tuple

import numpy as np

from phase2_model_training.preprocessing.median_beat import LEAD_II_INDEX, detect_r_peaks


def extract_rr_sequence(
    ecg_12lead: np.ndarray,
    fs: int = 500,
    max_len: int = 32,
    ref_lead_index: int = LEAD_II_INDEX,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """RR interval sequence(초 단위)와 padding mask를 만든다.

    Args:
        ecg_12lead: (12, T)
        fs: sampling rate
        max_len: 시퀀스 최대 길이(패딩/자르기 기준). PTB-XL 10초 레코딩은
                 보통 8~15개의 RR interval을 가지므로 32면 충분히 여유롭다.
        ref_lead_index: R-peak 검출에 쓸 lead (median_beat.py와 동일한 것을
                        써야 두 branch가 같은 beat 정의를 공유한다)

    Returns:
        rr_seq: (max_len,) RR interval(초). 유효하지 않은 위치는 0으로 패딩.
        mask: (max_len,) 1=유효, 0=패딩. Transformer/GRU의 padding mask로 사용.
        n_valid: 실제 유효한 RR interval 개수(0이면 beat을 2개 미만 검출 -> fallback 필요)
    """
    r_peaks = detect_r_peaks(ecg_12lead[ref_lead_index], fs=fs)

    if len(r_peaks) < 2:
        # RR interval을 하나도 못 만듦 (beat이 0~1개) -> 전부 패딩
        return (
            np.zeros(max_len, dtype=np.float32),
            np.zeros(max_len, dtype=np.float32),
            0,
        )

    rr_samples = np.diff(r_peaks)  # (n_beats-1,)
    rr_seconds = rr_samples / fs

    n_valid = min(len(rr_seconds), max_len)
    rr_seq = np.zeros(max_len, dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)
    rr_seq[:n_valid] = rr_seconds[:n_valid]
    mask[:n_valid] = 1.0

    return rr_seq, mask, n_valid


def rr_summary_stats(rr_seq: np.ndarray, mask: np.ndarray) -> dict:
    """디버깅/EDA용: RR sequence에서 흔히 쓰는 HRV 요약 통계.

    학습에 직접 쓰이진 않지만, extract_rr_sequence의 출력이 말이 되는지
    (예: mean RR이 생리학적으로 그럴듯한지) 확인할 때 유용하다.
    """
    valid = rr_seq[mask > 0]
    if len(valid) == 0:
        return dict(mean_rr=float("nan"), hr_bpm=float("nan"), sdnn=float("nan"), n=0)
    mean_rr = float(np.mean(valid))
    return dict(
        mean_rr=mean_rr,
        hr_bpm=60.0 / mean_rr if mean_rr > 0 else float("nan"),
        sdnn=float(np.std(valid)),
        n=len(valid),
    )
