"""
Soft-target Supervised Contrastive Loss.

s(i,j) (label_similarity.py에서 계산)가 0/1이면 정확히 vanilla SupCon으로
환원되는 일반화된 버전. 배치 내 각 anchor i에 대해, s(i,:)를 정규화한
분포를 target으로 삼아 InfoNCE의 log-softmax와의 cross entropy를 최소화한다.

수식은 phase2_model_training/losses/ 개발 중 numpy로 먼저 프로토타입해서
"라벨이 유사한 쌍을 임베딩에서 가깝게 배치하면 loss가 낮아진다"는 것을
합성 데이터로 검증한 뒤 그대로 옮긴 것이다.
"""
import torch
import torch.nn.functional as F


def soft_supcon_loss(
    z: torch.Tensor,
    s: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        z: (B, D) embedding (정규화 여부 상관없이 함수 내부에서 L2 정규화함)
        s: (B, B) soft target similarity matrix (label_similarity.py의 출력).
           대각선(self-similarity) 값은 사용하지 않음(내부에서 0으로 마스킹).
        temperature: InfoNCE temperature

    Returns:
        scalar loss. 배치 내에 target이 전부 0인 anchor(=positive가 하나도
        없는, 라벨 조합이 유니크한 샘플)는 자동으로 loss 계산에서 제외된다.
        모든 anchor가 그런 경우(예: batch_size=1) 0을 반환한다.
    """
    B = z.shape[0]
    z = F.normalize(z, dim=1, eps=eps)
    sim = torch.matmul(z, z.T) / temperature  # (B, B)

    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)
    # 대각선은 target에서도 0으로 마스킹되지만, log_prob의 대각선이 -inf이므로
    # target(0) * log_prob(-inf) = NaN이 되어 행 전체에 전파된다. 곱셈 전에
    # 대각선을 유한값(0)으로 만들어 NaN 전파를 막는다.
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    s = s.clone()
    s = s.masked_fill(self_mask, 0.0)
    row_sum = s.sum(dim=1, keepdim=True)  # (B, 1)
    valid = (row_sum.squeeze(-1) > eps)  # (B,)

    target = torch.zeros_like(s)
    if valid.any():
        target[valid] = s[valid] / row_sum[valid]

    per_anchor_loss = -(target * log_prob).sum(dim=1)  # (B,)

    if valid.sum() == 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return per_anchor_loss[valid].mean()
