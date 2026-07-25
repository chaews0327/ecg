"""
A0: Monolithic baseline.

Raw 12-lead ECG 전체를 하나의 1D ViT encoder에 넣어 하나의 embedding z를 만든다.
Loss는 form label과 rhythm label을 합친(concat) multi-hot vector로 하나의
soft-target SupCon을 건다 -- "지금까지의 MERL/MELP류 monolithic 모델을
동일한 학습 파이프라인/데이터로 재현한 baseline"에 해당한다.

인코더 하이퍼파라미터 기본값(width=768, layers=12, patch_size=50, head_width=64)은
업로드된 config.json의 ecg_cfg와 동일하게 맞췄다 -- 실제 프로덕션에서 쓰던
스케일을 baseline에도 그대로 반영.
"""
import torch.nn as nn

from phase2_model_training.models.encoders import CLIPEcgCfg, build_ecg_encoder


class A0Monolithic(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        embed_dim: int = 512,
        seq_length: int = 5000,
        width: int = 768,
        layers: int = 12,
        head_width: int = 64,
        patch_size: int = 50,
        mlp_ratio: float = 4.0,
        ls_init_value: float = 1e-5,   # <- 추가: 깊은 12-layer 모델일수록 이게 없으면 거의 collapse
    ):
        super().__init__()
        cfg = CLIPEcgCfg(
            layers=layers, width=width, head_width=head_width, mlp_ratio=mlp_ratio,
            patch_size=patch_size, seq_length=seq_length, lead_num=n_leads,
            ls_init_value=ls_init_value,
        )
        self.encoder = build_ecg_encoder(embed_dim, cfg)

    def forward(self, batch: dict) -> dict:
        z = self.encoder(batch["ecg"])  # (B, 12, 5000) -> (B, embed_dim)
        return {"z": z}
