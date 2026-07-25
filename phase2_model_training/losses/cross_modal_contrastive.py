"""
Cross-modal instance-level contrastive loss (CLIP 스타일).

soft_supcon_loss(라벨이 비슷한 샘플끼리 당김)와는 성격이 다르다. 여기서는
"라벨"이 아니라 "같은 ECG에서 나왔는가"가 positive를 정의한다:
  - z_full[i] (raw ECG 전체에서 나온 embedding)와 z_form[i](같은 ECG의 form
    branch embedding)는 positive pair
  - 배치 내 다른 인덱스 j != i의 z_form[j]는 전부 negative

이렇게 하면 "z_form이 원래 ECG(raw)와 여전히 연결되어 있는가"를 명시적으로
강제한다 -- form/rhythm으로 쪼개면서 원본 신호와의 연결이 끊어지는 것을
막는 두 번째 안전장치다(reconstruction auxiliary loss와 목적은 비슷하지만,
reconstruction은 "복원 가능한가", 이건 "같은 것이라고 식별 가능한가"를 본다).
"""
import torch
import torch.nn.functional as F


def cross_modal_info_nce(z_a: torch.Tensor, z_b: torch.Tensor,
                          temperature: float = 0.1) -> torch.Tensor:
    """
    Args:
        z_a, z_b: (B, D) 두 embedding. z_a[i]와 z_b[i]가 같은 ECG에서 나온
                  것이어야 한다(positive pair). D가 서로 달라도 되는 건 아니고
                  같은 embed_dim이어야 함(내적을 계산하므로).
        temperature: InfoNCE temperature

    Returns:
        symmetric InfoNCE loss (a->b 방향과 b->a 방향의 평균).
        CLIP 원 논문과 동일한 형태 -- 한 방향만 쓰면 z_a가 z_b 쪽으로만
        끌려가고 그 반대는 보장되지 않으므로 항상 대칭으로 계산한다.
    """
    B = z_a.shape[0]
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    logits = torch.matmul(z_a, z_b.T) / temperature  # (B, B)
    labels = torch.arange(B, device=z_a.device)       # 대각선(i,i)이 정답(positive)

    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    return (loss_a2b + loss_b2a) / 2