"""
A1: 2-branch, 동일 raw 입력, loss만 분리.

A0과 동일한 ECGEncoder(1D ViT) 구조를 2개 독립 인스턴스로 만들고(가중치는
독립), 둘 다 동일한 raw 12-lead ECG를 입력받는다. 차이는 오직 loss뿐이다
(z_form은 form label로, z_rhythm은 rhythm label로 각각 SupCon).

width/layers를 A0(768/12)보다 낮춘 이유: branch가 2개라 전체 파라미터 수가
A0의 2배 이상으로 커지므로, 동일한 컴퓨팅 예산 안에서 학습 가능하도록
스케일을 낮췄다. A0/A1을 "완전히 동일한 크기"로 맞추는 것이 목적이 아니라,
"동일 raw 입력을 두 branch로 나눴을 때"의 효과를 보는 것이 A1의 존재
이유이므로, 절대적 모델 크기보다 A1 vs A2(동일 크기)의 비교가 더 중요하다.

이 모델의 존재 이유는 "grain matching(A2)의 효과"를 "단순히 2개로 나눈 것"과
분리해서 측정하기 위함이다 -- A1 vs A2 성능 차이가 이 연구의 핵심 주장이다.
"""
import torch.nn as nn

from phase2_model_training.models.encoders import CLIPEcgCfg, build_ecg_encoder


class A1DualRaw(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        embed_dim: int = 256,
        seq_length: int = 5000,
        width: int = 384,
        layers: int = 6,
        head_width: int = 64,
        patch_size: int = 50,
        mlp_ratio: float = 4.0,
        ls_init_value: float = 1e-5,
    ):
        super().__init__()
        cfg = CLIPEcgCfg(
            layers=layers, width=width, head_width=head_width, mlp_ratio=mlp_ratio,
            patch_size=patch_size, seq_length=seq_length, lead_num=n_leads,
            ls_init_value=ls_init_value,
        )
        # A0와 동일한 구조를 두 번 인스턴스화 (가중치 공유 없음)
        self.form_encoder = build_ecg_encoder(embed_dim, cfg)
        self.rhythm_encoder = build_ecg_encoder(embed_dim, cfg)

    def forward(self, batch: dict) -> dict:
        ecg = batch["ecg"]  # 두 encoder 모두 동일한 raw 입력을 받음 (A2와의 핵심 차이)
        z_form = self.form_encoder(ecg)
        z_rhythm = self.rhythm_encoder(ecg)
        return {"z_form": z_form, "z_rhythm": z_rhythm}
