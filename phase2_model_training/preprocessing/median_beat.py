"""
R-peak 검출 + lead별 median(template) beat 추출.

간이 Pan-Tompkins 알고리즘(bandpass -> derivative -> square -> moving
integration -> adaptive peak picking)으로 reference lead(기본: lead II)에서
R-peak을 찾고, 그 위치를 기준으로 12-lead 전부에서 beat를 잘라 정렬한 뒤
sample-wise median을 취해 노이즈가 적은 대표 파형(template beat)을 만든다.

모든 lead가 동시 촬영이므로, reference lead에서 찾은 R-peak 위치를
다른 lead에도 그대로 적용할 수 있다(문헌에서도 흔히 쓰는 방식).
"""
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


LEAD_II_INDEX = 1  # 12-lead 표준 순서(I, II, III, aVR, aVL, aVF, V1-V6)에서 lead II


def detect_r_peaks(
    signal_1d: np.ndarray,
    fs: int = 500,
    ref_min_hr: float = 30.0,
    ref_max_hr: float = 220.0,
) -> np.ndarray:
    """단일 lead(1D) 신호에서 R-peak sample index를 검출한다 (간이 Pan-Tompkins).

    Args:
        signal_1d: (T,) 1-lead ECG
        fs: sampling rate
        ref_min_hr, ref_max_hr: 생리학적으로 가능한 심박수 범위(refractory
            period 및 이상치 제거에 사용)

    Returns:
        r_peaks: (n_beats,) 정렬된 R-peak sample index. 신호가 너무 짧거나
                 peak을 하나도 못 찾으면 빈 배열을 반환한다(호출부에서 처리).
    """
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    if len(signal_1d) < fs:  # 1초 미만은 처리하지 않음
        return np.array([], dtype=int)

    # 1) bandpass filter (5-15Hz, QRS complex 대역)
    nyq = fs / 2.0
    low, high = 5.0 / nyq, min(15.0 / nyq, 0.99)
    b, a = butter(N=2, Wn=[low, high], btype="band")
    filtered = filtfilt(b, a, signal_1d)

    # 2) derivative -> square -> moving window integration
    deriv = np.diff(filtered, prepend=filtered[0])
    squared = deriv**2
    win_len = max(1, int(0.150 * fs))  # 150ms 이동평균
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode="same")

    # 3) adaptive threshold + refractory period 기반 peak 검출
    min_distance = int(60.0 / ref_max_hr * fs)  # 최대 심박수 기준 refractory
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=max(1, min_distance), height=threshold)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # 4) 생리학적으로 불가능한 RR interval(너무 짧은 간격) 제거
    max_rr = fs * 60.0 / ref_min_hr
    min_rr = fs * 60.0 / ref_max_hr
    cleaned = [peaks[0]]
    for p in peaks[1:]:
        if (p - cleaned[-1]) >= min_rr:
            cleaned.append(p)
    r_peaks = np.array(cleaned, dtype=int)
    r_peaks = r_peaks[r_peaks < len(signal_1d)]
    return r_peaks


def extract_template_beat(
    ecg_12lead: np.ndarray,
    fs: int = 500,
    pre_r_ms: float = 200.0,
    post_r_ms: float = 400.0,
    min_beats: int = 3,
    ref_lead_index: int = LEAD_II_INDEX,
) -> Tuple[np.ndarray, int]:
    """12-lead ECG에서 lead별 median(template) beat를 추출한다.

    Args:
        ecg_12lead: (12, T) raw ECG
        fs: sampling rate
        pre_r_ms, post_r_ms: R-peak 기준 자를 구간(ms)
        min_beats: 이 개수 미만으로 beat가 검출되면 fallback으로 처리
        ref_lead_index: R-peak 검출에 쓸 reference lead (기본 lead II)

    Returns:
        template: (12, window_len) median beat. R-peak 검출 실패 시,
                  안전한 fallback으로 신호 앞부분을 window 길이만큼 잘라 반환한다
                  (학습이 죽지 않도록 하는 안전장치. 실제 배치에서는 이 fallback
                  비율을 반드시 로깅해서 monitoring할 것).
        n_beats_used: 실제로 median에 사용된 beat 개수 (0이면 fallback 사용됨)
    """
    ecg_12lead = np.asarray(ecg_12lead, dtype=np.float64)
    n_leads, T = ecg_12lead.shape
    pre = int(pre_r_ms / 1000 * fs)
    post = int(post_r_ms / 1000 * fs)
    window_len = pre + post

    r_peaks = detect_r_peaks(ecg_12lead[ref_lead_index], fs=fs)

    valid_peaks = [p for p in r_peaks if (p - pre) >= 0 and (p + post) <= T]

    if len(valid_peaks) < min_beats:
        # fallback: R-peak을 충분히 못 찾으면 신호 앞부분을 그대로 사용
        # (극단적으로 짧은/저품질 레코딩에 대한 안전장치)
        template = ecg_12lead[:, :window_len]
        if template.shape[1] < window_len:
            pad = window_len - template.shape[1]
            template = np.pad(template, ((0, 0), (0, pad)), mode="edge")
        return template.astype(np.float32), 0

    beats = np.stack(
        [ecg_12lead[:, p - pre : p + post] for p in valid_peaks], axis=0
    )  # (n_beats, 12, window_len)
    template = np.median(beats, axis=0)  # sample-wise median -> 노이즈에 강함
    return template.astype(np.float32), len(valid_peaks)


N_BEAT_VARIABILITY_FEATURES = 3


def extract_beat_variability_features(
    ecg_12lead: np.ndarray,
    fs: int = 500,
    pre_r_ms: float = 200.0,
    post_r_ms: float = 400.0,
    min_beats: int = 3,
    ref_lead_index: int = LEAD_II_INDEX,
) -> np.ndarray:
    """median beat가 평균내며 지워버리는 beat-to-beat 형태 변이를 요약한다.

    Template(median) beat 하나만으로는 "QRS가 매 beat마다 일관되게 넓은지"
    (LBBB), "ST elevation이 beat마다 일관되는지"(STEMI) 같이 여러 beat에
    걸쳐서만 드러나는 패턴을 form encoder가 볼 수 없다. R-peak 검출/beat
    정렬은 extract_template_beat와 동일한 로직(같은 reference lead)을 그대로
    반복한다 -- 두 함수가 다른 R-peak을 쓰면 template과 통계가 서로 다른
    beat 정의를 갖게 되어 앞뒤가 안 맞는다.

    Returns:
        (N_BEAT_VARIABILITY_FEATURES,) float32:
            [qrs_width_std, st_level_std, beat_shape_std]
        beat이 min_beats 미만이면 "변이를 측정할 근거 없음" 의미로 0벡터
        (extract_template_beat의 fallback과 동일한 경우에 대응).
    """
    ecg_12lead = np.asarray(ecg_12lead, dtype=np.float64)
    n_leads, T = ecg_12lead.shape
    pre = int(pre_r_ms / 1000 * fs)
    post = int(post_r_ms / 1000 * fs)

    r_peaks = detect_r_peaks(ecg_12lead[ref_lead_index], fs=fs)
    valid_peaks = [p for p in r_peaks if (p - pre) >= 0 and (p + post) <= T]

    if len(valid_peaks) < max(min_beats, 2):
        return np.zeros(N_BEAT_VARIABILITY_FEATURES, dtype=np.float32)

    beats = np.stack(
        [ecg_12lead[:, p - pre : p + post] for p in valid_peaks], axis=0
    )  # (n_beats, 12, window_len)

    # QRS width: R±100ms 구간에서 beat별 peak 진폭 대비 임계치를 넘는 구간의
    # 길이(초). LBBB처럼 QRS가 매 beat 일관되게 넓은지는 이 값의 std로 잡힌다
    # (median beat 자체는 폭의 '평균'만 보여주고 beat 간 일관성은 지운다).
    qrs_half = int(0.100 * fs)
    lo, hi = max(0, pre - qrs_half), pre + qrs_half
    qrs_seg = beats[:, ref_lead_index, lo:hi]
    thresh = 0.1 * np.max(np.abs(qrs_seg), axis=1, keepdims=True) + 1e-8
    qrs_width = (np.abs(qrs_seg) > thresh).sum(axis=1) / fs
    qrs_width_std = float(np.std(qrs_width))

    # ST level: R+80ms 지점의 진폭(전체 lead 평균). beat마다 이 값이 얼마나
    # 일관되는지가 STEMI의 "지속적 ST elevation" 여부와 관련된다.
    st_idx = min(pre + int(0.080 * fs), beats.shape[2] - 1)
    st_level = beats[:, :, st_idx].mean(axis=1)  # (n_beats,)
    st_level_std = float(np.std(st_level))

    # 전체 파형 변이: 시점/lead별 beat 간 std를 평균 -> beat들이 median에서
    # 얼마나 벗어나는지(형태 재현성 자체)에 대한 총괄 지표.
    beat_shape_std = float(np.mean(np.std(beats, axis=0)))

    return np.array([qrs_width_std, st_level_std, beat_shape_std], dtype=np.float32)
