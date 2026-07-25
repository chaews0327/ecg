"""
Soft supervised contrastive loss의 target similarity s(i,j) 계산.

기본은 Jaccard overlap(label_overlap)만 사용한다 -- 이것만으로도
hard 0/1 SupCon 대비 훨씬 나은 신호를 준다. Semantic similarity(scp_statements의
description을 clinical LM으로 임베딩해서 만드는 항)는 선택적으로 추가할 수
있도록 훅을 열어뒀다.

주의: scp_statements.csv의 description 컬럼 이름은 PTB-XL 버전에 따라
다를 수 있으므로("Statement", "SCP-ECG Statement Description" 등), semantic
similarity를 쓰려면 build_semantic_similarity_matrix()의 컬럼명을 본인
scp_statements.csv를 직접 열어 확인한 뒤 채워 넣어야 한다. 이 모듈은
그 값이 없어도(semantic_matrix=None) 정상 동작한다(overlap만 사용).
"""
from typing import Optional

import numpy as np


def label_overlap_similarity(y: np.ndarray) -> np.ndarray:
    """배치 내 모든 쌍에 대해 Jaccard overlap을 계산한다.

    Args:
        y: (B, C) multi-hot label matrix (0/1)

    Returns:
        s: (B, B) similarity matrix, s[i,i]는 사용하지 않으므로 0으로 둔다
           (loss 쪽에서 self-pair는 별도로 마스킹함)
    """
    y = (y > 0).astype(np.float64)
    intersection = y @ y.T  # (B,B)
    row_sum = y.sum(axis=1, keepdims=True)  # (B,1)
    union = row_sum + row_sum.T - intersection
    union = np.clip(union, a_min=1e-8, a_max=None)
    s = intersection / union
    np.fill_diagonal(s, 0.0)
    return s


def combine_similarity(
    overlap: np.ndarray,
    semantic: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """overlap과 (선택적) semantic similarity를 결합.

    semantic=None이면 overlap만 쓰는 것과 동일(alpha=1, beta=0 기본값과 일치).
    semantic similarity를 추가하려면 두 항목의 라벨 순서(컬럼 순서)가
    동일해야 한다.
    """
    if semantic is None:
        return np.clip(alpha * overlap, 0.0, 1.0)
    assert overlap.shape == semantic.shape, "overlap과 semantic similarity의 shape이 다릅니다"
    s = alpha * overlap + beta * semantic
    return np.clip(s, 0.0, 1.0)


def build_semantic_similarity_matrix(
    label_names: list,
    description_lookup: dict,
    text_encoder_fn,
) -> np.ndarray:
    """(선택 기능) 라벨 설명 텍스트 간 cosine similarity 행렬을 만든다.

    Args:
        label_names: 라벨(statement code) 이름 리스트, 길이 C
        description_lookup: {statement_code: description_text} 매핑.
                             scp_statements.csv에서 직접 만들어서 넘겨야 함.
        text_encoder_fn: 문자열 리스트 -> (C, D) 임베딩 반환하는 함수
                          (예: sentence-transformers 모델의 .encode)

    Returns:
        (C, C) cosine similarity 행렬. 이후 combine_similarity에 넘기기 전에
        배치의 (B,C) label matrix로부터 (B,B)로 변환하는 과정이 한 번 더
        필요하다 (label_to_pairwise_via_lookup 참고).
    """
    texts = [description_lookup.get(name, name) for name in label_names]
    emb = np.asarray(text_encoder_fn(texts))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb @ emb.T  # (C, C)


def build_cooccurrence_similarity_matrix(y_all: np.ndarray) -> np.ndarray:
    """(선택 기능, 외부 텍스트 인코더 불필요) 학습셋 전체에서의 라벨 co-occurrence로
    (C, C) semantic-ish similarity 행렬을 만든다.

    build_semantic_similarity_matrix()는 SCP-ECG statement의 설명 텍스트를
    clinical LM으로 임베딩해야 해서 외부 모델/컬럼명 확인이 필요했다. 이 함수는
    그 대신 "두 라벨이 데이터에서 얼마나 자주 함께 나타나는가"(예: AFIB와
    특정 conduction abnormality가 실제로 자주 동반되는 패턴)를 데이터 기반으로
    근사한 유사도로 쓴다. 외부 의존성이 없어 즉시 사용 가능하고, PTB-XL의
    실제 임상 공존 패턴을 반영한다는 점에서 나름의 타당성이 있다.

    Args:
        y_all: (N, C) 학습셋 전체의 multi-hot label matrix (배치가 아니라
               전체 train split을 한 번에 넣어서 미리 계산해두고 캐싱할 것)

    Returns:
        (C, C) similarity 행렬, 대각선은 1.0(자기 자신과는 완전히 유사).
        label_to_pairwise_via_lookup()에 그대로 넘기면 된다.
    """
    y_all = (y_all > 0).astype(np.float64)
    cooc = y_all.T @ y_all  # (C, C), cooc[a,b] = 라벨 a,b가 동시에 나타난 샘플 수
    freq = np.diag(cooc)  # 라벨별 등장 횟수
    denom = freq[:, None] + freq[None, :] - cooc  # union
    denom = np.clip(denom, 1e-8, None)
    sim = cooc / denom  # Jaccard 형태의 co-occurrence 유사도
    np.fill_diagonal(sim, 1.0)
    return sim


def label_to_pairwise_via_lookup(y: np.ndarray, class_similarity: np.ndarray) -> np.ndarray:
    """클래스 간 similarity(C,C)를 이용해, 배치 내 샘플 쌍(B,B)의 semantic
    similarity를 만든다. multi-label이므로, 두 샘플이 가진 라벨 집합 사이의
    최댓값 매칭을 평균하는 방식을 쓴다(각 라벨에 대해 상대방 라벨 중 가장
    가까운 것과 매칭).
    """
    y = (y > 0).astype(np.float64)
    B = y.shape[0]
    s = np.zeros((B, B))
    for i in range(B):
        labels_i = np.where(y[i] > 0)[0]
        if len(labels_i) == 0:
            continue
        for j in range(B):
            labels_j = np.where(y[j] > 0)[0]
            if len(labels_j) == 0:
                continue
            sub = class_similarity[np.ix_(labels_i, labels_j)]
            s[i, j] = sub.max(axis=1).mean()
    return s
