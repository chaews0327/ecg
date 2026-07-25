"""
Phase 2 모델들이 공유하는 encoder/decoder 아키텍처.

신규(이번 수정): 기존 ResNet1D를 1D ViT(ECGEncoder)로 전면 교체.
    ResidualBlock / TransformerEncoder / ECGEncoder / CLIPEcgCfg / build_ecg_encoder는
    사용자가 제공한 참고 코드(ecg_encoder.py, CLIP식 ECG encoder 구현)의 구조와
    이름을 그대로 이식한 것이다. 이후 CoCa류 멀티모달 코드와 연결이 필요해지면
    이름이 같아서 바로 재사용 가능하다.

- ECGEncoder: raw ECG(A0, A1) 및 template beat(A2 form branch) 둘 다에
  쓰는 1D ViT encoder. Conv1d로 patch를 나누고, CLS token + learnable
  positional embedding + Transformer encoder stack + linear projection으로
  고정 차원 embedding을 낸다.
- RRSequenceEncoder: RR interval sequence(A2 rhythm branch)를 위한 경량 GRU.
  ResNet이 아니므로 이번 교체 대상이 아니다. 그대로 유지.
- ConvDecoder1D / RRSequenceDecoder: A2의 reconstruction auxiliary loss용 decoder.
  encoder가 ResNet에서 ViT로 바뀌어도 "embed_dim 벡터 -> 원신호 복원"이라는
  역할은 동일하므로 수정 없이 그대로 재사용한다.

A0/A1/A2는 이 encoder/decoder들을 어떻게 조합하느냐의 차이일 뿐이며, 실제 정의는
models/a0_monolithic.py, a1_dual_raw.py, a2_grain_matched.py에 있다.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1D ViT (ecg_encoder.py 참고 코드를 이식)
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm Transformer block. cross-attention 경로(ln_1_kv)는 참고 코드의
    구조를 그대로 남겨뒀지만, 현재 A0/A1/A2 어디에서도 k/v를 별도로 넘기지
    않으므로(self-attention만 사용) is_cross_attention=False로만 쓰인다.
    나중에 CoCa 스타일로 text와 cross-attention이 필요해지면 그대로 활성화 가능."""

    def __init__(self, d_model, num_head, mlp_ratio, ls_init_value, act_layer, norm_layer,
                 is_cross_attention=False):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(self, q, k=None, v=None, attn_mask=None):
        # cross attention과 self attention이 함께 동작할 수 있도록
        k = k if k is not None else q
        v = v if v is not None else q
        attn_mask = attn_mask.to(q.dtype) if attn_mask is not None else None

        attn_output, attn_weights = self.attn(q, k, v, need_weights=True, attn_mask=attn_mask)
        return attn_output, attn_weights

    def forward(self, q, k=None, v=None, attn_mask=None):
        is_cross_attn = False

        if hasattr(self, "ln_1_kv"):
            is_cross_attn = True
            k = self.ln_1_kv(k)
            v = self.ln_1_kv(v)
        else:
            k = v = None

        attn_output, attn_weights = self.attention(self.ln_1(q), k, v, attn_mask)

        x = q + self.ls_1(attn_output)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))

        if is_cross_attn:
            return x, attn_weights

        return x


class TransformerEncoder(nn.Module):
    """ResidualBlock을 layers개 쌓은 스택. nn.MultiheadAttention이
    batch_first=False(기본값)이므로, (B, seq, width) <-> (seq, B, width)를
    앞뒤로 transpose한다(참고 코드와 동일한 방식)."""

    def __init__(self, width, layers, heads, mlp_ratio=4.0, ls_init_value=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            ResidualBlock(width, heads, mlp_ratio, ls_init_value, act_layer, norm_layer)
            for _ in range(layers)
        ])

    def forward(self, x, attn_mask=None):
        x = x.transpose(0, 1)

        for block in self.resblocks:
            x = block(x, attn_mask)
        x = x.transpose(0, 1)

        return x


class ECGEncoder(nn.Module):
    """1D ECG ViT. (B, lead_num, seq_length) -> (B, output_dim).

    seq_length가 patch_size로 나누어떨어지지 않으면(예: 배치마다 길이가
    조금씩 다른 경우) forward()에서 자동으로 pad/crop한다 -- 참고 코드에는
    없던 안전장치이지만, ResNet1D가 AdaptiveAvgPool로 가변 길이를 흡수하던
    것과 동등한 안정성을 ViT에서도 보장하기 위해 추가했다.
    """

    def __init__(self, seq_length, patch_size, lead_num, width, layers, heads, mlp_ratio, ls_init_value,
                 final_ln_after_pool=False, pool_type='tok', output_tokens=False, output_dim=512,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_tokens = output_tokens
        self.seq_length = seq_length  # 전체 길이 (factory.py의 get_model_preprocess_cfg가 참조하는 속성명과 동일하게 유지)
        self.lead_num = lead_num  # 채널
        self.patch_size = patch_size  # 패치 크기
        self.patch_nums = seq_length // patch_size  # 전체 패치 개수
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim  # 최종 차원

        self.conv1 = nn.Conv1d(
            in_channels=lead_num,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.patch_nums + 1, width))
        self.patch_dropout = nn.Identity()

        self.ln_pre = norm_layer(width)

        self.transformer = TransformerEncoder(width, layers, heads, mlp_ratio,
                                               ls_init_value=ls_init_value,
                                               act_layer=act_layer,
                                               norm_layer=norm_layer)

        pool_dim = width
        self.pool_type = pool_type  # 참고 코드와 동일하게 현재는 사용 X ('tok' 고정)

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter((pool_dim ** -0.5) * torch.randn(pool_dim, output_dim))

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lead_num, T). T가 seq_length와 다르면 seq_length에 맞춘다."""
        T = x.shape[-1]
        if T == self.seq_length:
            return x
        if T < self.seq_length:
            return F.pad(x, (0, self.seq_length - T))
        return x[..., :self.seq_length]

    def forward(self, x, output_last_transformer_layer=False):
        x = self._pad_or_crop(x)
        x = self.conv1(x)  # (*, width, num_patch)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # (*, num_patch, width)

        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)  # (*, num_patch+1, width), broadcasting으로 배치 차원에 더해짐
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if output_last_transformer_layer:
            return x

        x = self.ln_post(x)
        pooled, tokens = x[:, 0], x[:, 1:]

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])


@dataclass
class CLIPEcgCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 50
    seq_length: int = 5000
    lead_num: int = 12

    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None


def build_ecg_encoder(embed_dim, ecg_cfg):
    if isinstance(ecg_cfg, dict):
        ecg_cfg = CLIPEcgCfg(**ecg_cfg)
    ecg_heads = ecg_cfg.width // ecg_cfg.head_width

    ecg_model = ECGEncoder(
        seq_length=ecg_cfg.seq_length,
        patch_size=ecg_cfg.patch_size,
        lead_num=ecg_cfg.lead_num,
        width=ecg_cfg.width,
        layers=ecg_cfg.layers,
        heads=ecg_heads,
        mlp_ratio=ecg_cfg.mlp_ratio,
        ls_init_value=ecg_cfg.ls_init_value,
        final_ln_after_pool=ecg_cfg.final_ln_after_pool,
        pool_type=ecg_cfg.pool_type,
        output_tokens=ecg_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )

    return ecg_model


# ----------------------------------------------------------------------
# Rhythm branch: RR interval sequence encoder (GRU, ResNet 아님 -> 변경 없음)
# ----------------------------------------------------------------------

class RRSequenceEncoder(nn.Module):
    """RR interval sequence(가변 길이, padding mask 포함) -> (B, embed_dim).

    입력은 (B, L) 스칼라 RR interval 값(초 단위)이며, 여기에 작은 linear
    projection으로 차원을 올린 뒤 GRU에 태운다. Beat 하나하나의 형태가
    아니라 "beat 간 간격의 패턴"만 보는 것이 grain-matching의 핵심이므로,
    입력을 의도적으로 RR interval 스칼라 시퀀스로 제한한다.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, rr_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(rr_seq.unsqueeze(-1))  # (B, L, hidden_dim)
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: (num_layers*2, B, hidden_dim)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=-1)  # (B, hidden_dim*2)
        return self.proj(h)


# ----------------------------------------------------------------------
# Reconstruction decoder (A2, encoder 종류와 무관하게 그대로 재사용)
# ----------------------------------------------------------------------

class ConvDecoder1D(nn.Module):
    """(B, embed_dim) -> (B, n_channels, target_length) 복원 decoder.

    F.interpolate로 원하는 target_length에 항상 정확히 도달하도록 만들어,
    encoder가 ResNet이든 ViT든 상관없이 그대로 재사용 가능하다.
    """

    def __init__(self, embed_dim: int, n_channels: int, target_length: int,
                 hidden: int = 128, init_len: int = 8):
        super().__init__()
        self.hidden = hidden
        self.init_len = init_len
        self.target_length = target_length
        self.fc = nn.Linear(embed_dim, hidden * init_len)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden // 2, n_channels, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.fc(z).view(B, self.hidden, self.init_len)
        h = F.interpolate(h, size=self.target_length, mode="linear", align_corners=False)
        return self.net(h)


class RRSequenceDecoder(nn.Module):
    """(B, embed_dim) -> (B, max_len) RR interval sequence 복원."""

    def __init__(self, embed_dim: int, max_len: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     eps: float = 1.0) -> torch.Tensor:
    """padding(mask=0)된 위치는 무시하는 MSE."""
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=eps)
    return diff2.sum() / denom

class LayerScale(nn.Module):
    """residual branch의 출력에 학습 가능한 작은 스케일(gamma)을 곱해서,
    초반 학습에서 residual stream이 폭주/collapse하지 않도록 억제한다
    (CaiT/DeiT의 LayerScale과 동일한 목적). ls_init_value가 매우 작은 값
    (1e-5 근처)이면 학습 초반엔 residual block이 거의 identity처럼 동작하다가,
    학습이 진행되며 gamma가 점점 커지면서 해당 block의 기여가 살아난다."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma