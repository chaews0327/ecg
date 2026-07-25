"""
A2: Grain-matched dual encoder (제안 방법) + reconstruction + cross-modal contrastive.

A1과 구조(1D ViT encoder)는 유사하지만, **입력 자체가 다르다**:
  - form_encoder: lead별 median/template beat (12, ~300) + beat-to-beat
    형태 변이 통계(qrs_width_std, st_level_std, beat_shape_std)를 결합.
    template beat 길이가 300으로 raw ECG(5000)보다 훨씬 짧으므로, patch_size와
    width를 raw ECG를 보는 encoder(A1, A2의 full_encoder)보다 가볍게 줬다
    (width 256, layers 6, patch_size 20 -> patch 15개).
  - rhythm_encoder: RR interval sequence -- 리듬은 beat 간의 관계에서
    정의된다는 원칙을 반영. ViT가 아니라 RRSequenceEncoder(GRU)를 그대로 씀
    (RR sequence는 스칼라 시계열이라 patch 기반 ViT보다 GRU가 더 자연스럽다).
  - full_encoder(use_full_ecg_branch=True일 때): raw ECG 전체를 보는 세 번째
    encoder. A1의 raw encoder와 동일한 스케일(width 384, layers 6)을 쓴다.
    z_full 자체는 최종 산출물이 아니라 cross-modal contrastive의 "닻" 역할.

A1과 A2의 유일한 구조적 차이는 "입력 grain"이므로, 두 모델의 성능/leakage
차이가 곧 grain matching 자체의 순수한 기여도가 된다.
"""
import torch
import torch.nn as nn

from phase2_model_training.models.encoders import (
    CLIPEcgCfg,
    ConvDecoder1D,
    RRSequenceDecoder,
    RRSequenceEncoder,
    build_ecg_encoder,
)
from phase2_model_training.preprocessing.median_beat import N_BEAT_VARIABILITY_FEATURES


class A2GrainMatched(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        embed_dim: int = 256,
        n_beat_stats: int = N_BEAT_VARIABILITY_FEATURES,
        template_beat_length: int = 300,
        rr_max_len: int = 32,
        use_reconstruction: bool = True,
        use_full_ecg_branch: bool = True,
        # form branch (template beat) ViT 설정
        form_width: int = 256,
        form_layers: int = 6,
        form_patch_size: int = 20,
        form_head_width: int = 64,
        # full ECG branch(cross-modal 용) ViT 설정 -- A1의 raw encoder와 동일 스케일
        full_seq_length: int = 5000,
        full_width: int = 384,
        full_layers: int = 6,
        full_patch_size: int = 50,
        full_head_width: int = 64,
        mlp_ratio: float = 4.0,
        ls_init_value: float = 1e-5, 
    ):
        super().__init__()
        self.use_reconstruction = use_reconstruction
        self.use_full_ecg_branch = use_full_ecg_branch

        form_cfg = CLIPEcgCfg(
            layers=form_layers, width=form_width, head_width=form_head_width,
            mlp_ratio=mlp_ratio, patch_size=form_patch_size,
            seq_length=template_beat_length, lead_num=n_leads,
            ls_init_value=ls_init_value,
        )
        self.form_encoder = build_ecg_encoder(embed_dim, form_cfg)

        self.form_stats_proj = nn.Sequential(
            nn.Linear(n_beat_stats, 32),
            nn.ReLU(inplace=True),
        )
        self.form_combine = nn.Linear(embed_dim + 32, embed_dim)
        self.rhythm_encoder = RRSequenceEncoder(embed_dim=embed_dim)

        if use_reconstruction:
            self.form_decoder = ConvDecoder1D(
                embed_dim=embed_dim, n_channels=n_leads, target_length=template_beat_length
            )
            self.rhythm_decoder = RRSequenceDecoder(embed_dim=embed_dim, max_len=rr_max_len)

        if use_full_ecg_branch:
            full_cfg = CLIPEcgCfg(
                layers=full_layers, width=full_width, head_width=full_head_width,
                mlp_ratio=mlp_ratio, patch_size=full_patch_size,
                seq_length=full_seq_length, lead_num=n_leads,
            )
            self.full_encoder = build_ecg_encoder(embed_dim, full_cfg)

    def forward(self, batch: dict) -> dict:
        h_template = self.form_encoder(batch["template_beat"])       # (B, 12, ~300) -> (B, embed_dim)
        h_stats = self.form_stats_proj(batch["beat_stats"])           # (B, 3) -> (B, 32)
        z_form = self.form_combine(torch.cat([h_template, h_stats], dim=1))
        z_rhythm = self.rhythm_encoder(batch["rr_seq"], batch["rr_mask"])  # (B, L), (B, L)

        out = {"z_form": z_form, "z_rhythm": z_rhythm}

        if self.use_reconstruction:
            out["recon_template_beat"] = self.form_decoder(z_form)     # (B, 12, ~300)
            out["recon_rr_seq"] = self.rhythm_decoder(z_rhythm)        # (B, L)

        if self.use_full_ecg_branch:
            out["z_full"] = self.full_encoder(batch["ecg"])            # (B, 12, 5000) -> (B, embed_dim)

        return out
