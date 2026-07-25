"""
Phase 2 학습 스크립트: A0 / A1 / A2 공용.

신규(이번 수정):
    - A2에 reconstruction auxiliary loss 결합 (lambda_recon으로 가중치 조절).
      L_total = L_form_supcon + L_rhythm_supcon + lambda_recon * (L_recon_form + L_recon_rhythm)
    - --use_cooccurrence_similarity 옵션: label overlap(Jaccard)뿐 아니라, 학습셋
      전체의 라벨 co-occurrence 통계로 만든 semantic-ish similarity를 결합해서
      soft target을 더 풍부하게 만듦 (label_similarity.build_cooccurrence_similarity_matrix).
    - --no_reconstruction 옵션으로 A2에서 reconstruction만 뺀 A3 ablation도 바로 재현 가능.

사용법:
    # 실제 데이터 없이 배선만 확인 (모델 forward + loss + backward가 도는지)
    python -m phase2_model_training.train_phase2 --model a0 --sanity_check
    python -m phase2_model_training.train_phase2 --model a1 --sanity_check
    python -m phase2_model_training.train_phase2 --model a2 --sanity_check

    # 실제 학습
    python -m phase2_model_training.train_phase2 --model a0 --epochs 100
    python -m phase2_model_training.train_phase2 --model a1 --epochs 100
    python -m phase2_model_training.train_phase2 --model a2 --epochs 100 --lambda_recon 0.5

    # A3 ablation (A2에서 reconstruction만 제거)
    python -m phase2_model_training.train_phase2 --model a2 --epochs 100 --no_reconstruction

    # semantic(co-occurrence) similarity까지 결합
    python -m phase2_model_training.train_phase2 --model a2 --epochs 100 --use_cooccurrence_similarity
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

import config
from data.ptbxl_dataset import make_unified_dataset
from phase2_model_training.losses.label_similarity import (
    build_cooccurrence_similarity_matrix,
    combine_similarity,
    label_overlap_similarity,
    label_to_pairwise_via_lookup,
)
from phase2_model_training.losses.soft_supcon import soft_supcon_loss
from phase2_model_training.models.a0_monolithic import A0Monolithic
from phase2_model_training.models.a1_dual_raw import A1DualRaw
from phase2_model_training.models.a2_grain_matched import A2GrainMatched
from phase2_model_training.models.encoders import masked_mse_loss
from phase2_model_training.losses.cross_modal_contrastive import cross_modal_info_nce
from phase2_model_training.preprocessing.phase2_dataset import make_grain_matched_dataset


MODEL_REGISTRY = {"a0": A0Monolithic, "a1": A1DualRaw, "a2": A2GrainMatched}


def build_model(name: str, use_reconstruction: bool = True,
                 use_full_ecg_branch: bool = True) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 모델: {name}. {list(MODEL_REGISTRY.keys())} 중 하나여야 함")
    if name == "a2":
        return A2GrainMatched(
            use_reconstruction=use_reconstruction,
            use_full_ecg_branch=use_full_ecg_branch,
        )
    return MODEL_REGISTRY[name]()


# ----------------------------------------------------------------------
# Similarity 계산: overlap(+선택적 co-occurrence semantic) -> (B,B) soft target
# ----------------------------------------------------------------------

class SimilarityComputer:
    """배치 라벨 -> (B,B) soft target similarity.

    use_cooccurrence=True면, 생성 시점에 넘겨준 학습셋 전체 라벨(y_all_train)로
    (C,C) class-level co-occurrence similarity를 한 번만 계산해 캐싱해두고,
    매 배치마다 label_to_pairwise_via_lookup으로 (B,B)로 변환해 overlap과 결합한다.
    """

    def __init__(self, use_cooccurrence: bool = False,
                 y_all_train_form: np.ndarray = None, y_all_train_rhythm: np.ndarray = None,
                 alpha: float = 0.7, beta: float = 0.3):
        self.use_cooccurrence = use_cooccurrence
        self.alpha = alpha
        self.beta = beta
        self.class_sim_form = None
        self.class_sim_rhythm = None
        if use_cooccurrence:
            assert y_all_train_form is not None and y_all_train_rhythm is not None, (
                "use_cooccurrence=True면 학습셋 전체 라벨을 미리 넘겨야 class-level "
                "co-occurrence 행렬을 계산할 수 있습니다."
            )
            self.class_sim_form = build_cooccurrence_similarity_matrix(y_all_train_form)
            self.class_sim_rhythm = build_cooccurrence_similarity_matrix(y_all_train_rhythm)

    def __call__(self, y: torch.Tensor, task: str) -> torch.Tensor:
        """y: (B, C) label tensor, task: 'form' 또는 'rhythm' (어떤 class_sim을 쓸지 결정)"""
        y_np = y.detach().cpu().numpy()
        overlap = label_overlap_similarity(y_np)

        if self.use_cooccurrence:
            class_sim = self.class_sim_form if task == "form" else self.class_sim_rhythm
            semantic = label_to_pairwise_via_lookup(y_np, class_sim)
            s_np = combine_similarity(overlap, semantic, alpha=self.alpha, beta=self.beta)
        else:
            s_np = combine_similarity(overlap)  # overlap만 사용 (alpha=1, beta=0 기본값)

        return torch.from_numpy(s_np).to(y.device, dtype=torch.float32)
    
    
def build_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                                   base_lr: float, eta_min: float = 1e-6):
    """
    epoch 0 ~ warmup_epochs: lr이 0 -> base_lr로 선형 증가 (warmup)
    epoch warmup_epochs ~ total_epochs: lr이 base_lr -> eta_min으로 cosine 감소

    LambdaLR은 "배수(multiplier)"를 돌려주는 방식이라, 절대 lr이 아니라
    base_lr 대비 비율로 계산한다. eta_min_ratio = eta_min / base_lr.
    """
    eta_min_ratio = eta_min / base_lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # 선형 warmup: epoch 0에서 1/warmup_epochs, epoch warmup_epochs-1에서 거의 1.0
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            # cosine decay: warmup이 끝난 시점을 progress=0으로 재설정
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            progress = min(progress, 1.0)  # 혹시 epoch이 total_epochs를 넘어도 안전하게
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return eta_min_ratio + (1 - eta_min_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def training_step(
    model_name: str,
    model: torch.nn.Module,
    batch: dict,
    similarity_fn: SimilarityComputer,
    temperature: float = 0.1,
    lambda_recon: float = 0.5,
    lambda_cross_modal: float = 0.3,   # <- 신규
) -> dict:
    if model_name == "a0":
        out = model(batch)
        y_combined = torch.cat([batch["form_label"], batch["rhythm_label"]], dim=1)
        y_np = y_combined.detach().cpu().numpy()
        s_np = combine_similarity(label_overlap_similarity(y_np))
        s = torch.from_numpy(s_np).to(y_combined.device, dtype=torch.float32)
        loss = soft_supcon_loss(out["z"], s, temperature=temperature)
        return {"total": loss, "form": loss.detach(), "rhythm": loss.detach()}

    elif model_name in ("a1", "a2"):
        out = model(batch)
        s_form = similarity_fn(batch["form_label"], task="form")
        s_rhythm = similarity_fn(batch["rhythm_label"], task="rhythm")
        loss_form = soft_supcon_loss(out["z_form"], s_form, temperature=temperature)
        loss_rhythm = soft_supcon_loss(out["z_rhythm"], s_rhythm, temperature=temperature)
        total = loss_form + loss_rhythm

        result = {"form": loss_form.detach(), "rhythm": loss_rhythm.detach()}

        if model_name == "a2" and "recon_template_beat" in out:
            recon_loss_form = torch.nn.functional.mse_loss(
                out["recon_template_beat"], batch["template_beat"]
            )
            recon_loss_rhythm = masked_mse_loss(
                out["recon_rr_seq"], batch["rr_seq"], batch["rr_mask"]
            )
            total = total + lambda_recon * (recon_loss_form + recon_loss_rhythm)
            result["recon_form"] = recon_loss_form.detach()
            result["recon_rhythm"] = recon_loss_rhythm.detach()

        # --- 신규: 같은 ECG의 z_full과 z_form/z_rhythm을 positive로 정렬 ---
        if model_name == "a2" and "z_full" in out:
            cross_form = cross_modal_info_nce(out["z_full"], out["z_form"], temperature=temperature)
            cross_rhythm = cross_modal_info_nce(out["z_full"], out["z_rhythm"], temperature=temperature)
            total = total + lambda_cross_modal * (cross_form + cross_rhythm)
            result["cross_modal_form"] = cross_form.detach()
            result["cross_modal_rhythm"] = cross_rhythm.detach()
        # --- ---

        result["total"] = total
        return result

    else:
        raise ValueError(model_name)


def make_dataloader(model_name: str, split: str, batch_size: int, num_workers: int) -> DataLoader:
    if model_name in ("a0", "a1"):
        dataset = make_unified_dataset(split=split, ptbxl_root=config.PTBXL_ROOT, split_dir=config.SPLIT_DIR)
    elif model_name == "a2":
        dataset = make_grain_matched_dataset(split=split, ptbxl_root=config.PTBXL_ROOT, split_dir=config.SPLIT_DIR)
    else:
        raise ValueError(model_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)


def build_similarity_computer(model_name: str, use_cooccurrence: bool, train_loader: DataLoader) -> SimilarityComputer:
    """use_cooccurrence=True면 train_loader 전체를 한 번 순회해서 라벨을 모으고
    class-level co-occurrence 행렬을 미리 계산한다 (매 배치 재계산 방지)."""
    if not use_cooccurrence or model_name == "a0":
        return SimilarityComputer(use_cooccurrence=False)

    form_labels, rhythm_labels = [], []
    for batch in train_loader:
        form_labels.append(batch["form_label"].numpy())
        rhythm_labels.append(batch["rhythm_label"].numpy())
    y_all_form = np.concatenate(form_labels, axis=0)
    y_all_rhythm = np.concatenate(rhythm_labels, axis=0)
    print(f"[co-occurrence similarity] train 전체 {len(y_all_form)}개 샘플로 "
          f"class-level similarity 행렬 계산 완료 (form {y_all_form.shape[1]}x{y_all_form.shape[1]}, "
          f"rhythm {y_all_rhythm.shape[1]}x{y_all_rhythm.shape[1]})")
    return SimilarityComputer(use_cooccurrence=True, y_all_train_form=y_all_form, y_all_train_rhythm=y_all_rhythm)


# ----------------------------------------------------------------------
# Sanity check: 실제 데이터 없이 모델 forward + loss + backward가 도는지 확인
# ----------------------------------------------------------------------

def _make_fake_batch(model_name: str, batch_size: int = 16, device: str = "cpu",
                      use_full_ecg_branch: bool = True) -> dict:
    n_form, n_rhythm, n_beat_stats = 19, 12, 3
    form_label = (torch.rand(batch_size, n_form) > 0.85).float()
    rhythm_label = (torch.rand(batch_size, n_rhythm) > 0.85).float()
    # 라벨이 전부 0인 행이 있으면 label_overlap_similarity에서 0/0 나눗셈 위험 -> 최소 1개는 보장
    form_label[:, 0] = 1.0
    rhythm_label[:, 0] = 1.0

    batch = {"form_label": form_label.to(device), "rhythm_label": rhythm_label.to(device)}

    if model_name in ("a0", "a1"):
        batch["ecg"] = torch.randn(batch_size, 12, 5000, device=device)
    elif model_name == "a2":
        batch["template_beat"] = torch.randn(batch_size, 12, 300, device=device)
        batch["beat_stats"] = torch.randn(batch_size, n_beat_stats, device=device)
        rr_seq = torch.rand(batch_size, 32, device=device) * 0.3 + 0.6
        rr_mask = torch.zeros(batch_size, 32, device=device)
        for i in range(batch_size):
            n_valid = torch.randint(5, 15, (1,)).item()
            rr_mask[i, :n_valid] = 1.0
        batch["rr_seq"] = rr_seq
        batch["rr_mask"] = rr_mask
        if use_full_ecg_branch:
            batch["ecg"] = torch.randn(batch_size, 12, 5000, device=device)   # <- 추가
    return batch


def run_sanity_check(model_name: str, n_steps: int = 5, use_reconstruction: bool = True,
                      use_full_ecg_branch: bool = True):
    print(f"### [sanity check] model={model_name} "
          f"(reconstruction={use_reconstruction}, full_ecg_branch={use_full_ecg_branch}) ###")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, use_reconstruction=use_reconstruction,
                         use_full_ecg_branch=use_full_ecg_branch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    similarity_fn = SimilarityComputer(use_cooccurrence=False)

    losses = []
    for step in range(n_steps):
        batch = _make_fake_batch(model_name, batch_size=16, device=device,
                                  use_full_ecg_branch=use_full_ecg_branch)
        loss_dict = training_step(model_name, model, batch, similarity_fn)

        optimizer.zero_grad()
        loss_dict["total"].backward()   # <- backward는 total에 대해서만
        optimizer.step()

        total = loss_dict["total"].item()
        form = loss_dict["form"].item()
        rhythm = loss_dict["rhythm"].item()
        losses.append(total)

        log_line = f"  step {step}: total={total:.4f}  form={form:.4f}  rhythm={rhythm:.4f}"
        if "recon_form" in loss_dict:
            log_line += f"  recon_form={loss_dict['recon_form'].item():.4f}"
        if "recon_rhythm" in loss_dict:
            log_line += f"  recon_rhythm={loss_dict['recon_rhythm'].item():.4f}"
        print(log_line)

    assert all(np.isfinite(losses)), "loss에 NaN/Inf가 발생함 -- 아키텍처/loss 배선을 확인하세요"
    print(f"  통과: {n_steps} step 동안 NaN/Inf 없이 학습됨 (loss가 반드시 단조 감소할 필요는 없음, "
          f"랜덤 데이터이므로 finite 여부만 확인)\n")


# ----------------------------------------------------------------------
# 실제 학습
# ----------------------------------------------------------------------

def train(model_name: str, epochs: int, batch_size: int, lr: float, num_workers: int,
          temperature: float, lambda_recon: float, use_reconstruction: bool,
          use_cooccurrence_similarity: bool, patience: int = 10, min_delta: float = 1e-4,
          eta_min: float = 1e-6, warmup_epochs: int = 5,
          lambda_cross_modal: float = 0.3, use_full_ecg_branch: bool = True):  # <- 신규 2개
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = build_model(
        model_name,
        use_reconstruction=use_reconstruction,
        use_full_ecg_branch=use_full_ecg_branch,   # <- 전달
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs, lr, eta_min)

    train_loader = make_dataloader(model_name, "train", batch_size, num_workers)
    val_loader = make_dataloader(model_name, "val", batch_size, num_workers)
    similarity_fn = build_similarity_computer(model_name, use_cooccurrence_similarity, train_loader)

    ckpt_dir = config.OUTPUT_DIR / "phase2_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch_history = []

    for epoch in range(epochs):
        model.train()
        train_totals, train_forms, train_rhythms = [], [], []

        pbar = tqdm(train_loader, desc=f"[{model_name}] epoch {epoch} train")
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss_dict = training_step(
                model_name, model, batch, similarity_fn,
                temperature=temperature, lambda_recon=lambda_recon,
                lambda_cross_modal=lambda_cross_modal,   # <- 전달
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            total = loss_dict["total"].item()
            form = loss_dict["form"].item()
            rhythm = loss_dict["rhythm"].item()
            train_totals.append(total)
            train_forms.append(form)
            train_rhythms.append(rhythm)

            postfix = {"total": f"{total:.4f}", "form": f"{form:.4f}", "rhythm": f"{rhythm:.4f}"}
            if "recon_form" in loss_dict:
                postfix["recon_f"] = f"{loss_dict['recon_form'].item():.4f}"
                postfix["recon_r"] = f"{loss_dict['recon_rhythm'].item():.4f}"
            if "cross_modal_form" in loss_dict:                                    # <- 로깅에도 추가
                postfix["xm_f"] = f"{loss_dict['cross_modal_form'].item():.4f}"
                postfix["xm_r"] = f"{loss_dict['cross_modal_rhythm'].item():.4f}"
            pbar.set_postfix(postfix)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_totals, val_forms, val_rhythms = [], [], []
        val_pbar = tqdm(val_loader, desc=f"[{model_name}] epoch {epoch} val")
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss_dict = training_step(
                    model_name, model, batch, similarity_fn,
                    temperature=temperature, lambda_recon=lambda_recon,
                    lambda_cross_modal=lambda_cross_modal,   # <- 전달
                )
                total = loss_dict["total"].item()
                form = loss_dict["form"].item()
                rhythm = loss_dict["rhythm"].item()
                val_totals.append(total)
                val_forms.append(form)
                val_rhythms.append(rhythm)
                val_pbar.set_postfix({"total": f"{total:.4f}", "form": f"{form:.4f}", "rhythm": f"{rhythm:.4f}"})

        train_loss = float(np.mean(train_totals))
        val_loss = float(np.mean(val_totals))
        val_form_loss = float(np.mean(val_forms))
        val_rhythm_loss = float(np.mean(val_rhythms))

        print(f"[{model_name}] epoch {epoch}: train_total={train_loss:.4f}  "
              f"val_total={val_loss:.4f}  val_form={val_form_loss:.4f}  "
              f"val_rhythm={val_rhythm_loss:.4f}  lr={current_lr:.2e}")

        epoch_history.append({
            "epoch": epoch, "lr": current_lr,
            "train_total": train_loss, "val_total": val_loss,
            "val_form": val_form_loss, "val_rhythm": val_rhythm_loss,
        })

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_dir / f"{model_name}_best.pt")
            print(f"  -> best checkpoint 저장: {ckpt_dir / f'{model_name}_best.pt'}")
        else:
            epochs_without_improvement += 1
            print(f"  -> 개선 없음 ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"[{model_name}] epoch {epoch}: early stopping "
                      f"(best_val_loss={best_val_loss:.4f}).")
                break

    torch.save(model.state_dict(), ckpt_dir / f"{model_name}_last.pt")

    import json
    history_path = ckpt_dir / f"{model_name}_loss_history.json"
    with open(history_path, "w") as f:
        json.dump(epoch_history, f, indent=2)
    print(f"학습 종료. best/last checkpoint -> {ckpt_dir}")
    print(f"form/rhythm loss history 저장 -> {history_path}")


def main():
    DEFAULT_HYPERPARAMS = {
        "a0": {
            "lr": 2e-4, "batch_size": 256,
            "lambda_recon": 0.0, "lambda_cross_modal": 0.0,   # A0는 form/rhythm 분리가 없어 해당 없음
            "patience": 10, "min_delta": 1e-4,
            "warmup_epochs": 5, "eta_min": 1e-6,
        },
        "a1": {
            "lr": 2e-4, "batch_size": 256,
            "lambda_recon": 0.0, "lambda_cross_modal": 0.0,   # A1은 raw ECG를 그대로 쓰므로 해당 없음
            "patience": 10, "min_delta": 1e-4,
            "warmup_epochs": 5, "eta_min": 1e-6,
        },
        "a2": {
            "lr": 1e-4, "batch_size": 128,
            "lambda_recon": 0.5,
            "lambda_cross_modal": 0.3,   # <- 신규: 여기서 조정
            "patience": 15, "min_delta": 5e-5,
            "warmup_epochs": 10, "eta_min": 1e-7,
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_recon", type=float, default=None)
    parser.add_argument("--no_reconstruction", action="store_true")
    parser.add_argument("--use_cooccurrence_similarity", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--eta_min", type=float, default=None)

    # --- 신규 2개 ---
    parser.add_argument("--lambda_cross_modal", type=float, default=None,
                         help="A2 only. z_full <-> z_form/z_rhythm cross-modal contrastive 가중치. "
                              "기본값: 모델별 DEFAULT_HYPERPARAMS 사용")
    parser.add_argument("--no_full_ecg_branch", action="store_true",
                         help="A2에서 z_full(raw ECG) branch와 cross-modal loss를 통째로 끔 "
                              "(ablation: cross-modal 기여도만 따로 보고 싶을 때)")
    # --- ---

    args = parser.parse_args()

    use_reconstruction = not args.no_reconstruction
    use_full_ecg_branch = not args.no_full_ecg_branch   # <- 신규

    model_defaults = DEFAULT_HYPERPARAMS[args.model]

    lr = args.lr if args.lr is not None else model_defaults["lr"]
    batch_size = args.batch_size if args.batch_size is not None else model_defaults["batch_size"]
    patience = args.patience if args.patience is not None else model_defaults["patience"]
    min_delta = args.min_delta if args.min_delta is not None else model_defaults["min_delta"]
    lambda_recon = args.lambda_recon if args.lambda_recon is not None else model_defaults["lambda_recon"]
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else model_defaults["warmup_epochs"]
    eta_min = args.eta_min if args.eta_min is not None else model_defaults["eta_min"]
    lambda_cross_modal = (
        args.lambda_cross_modal if args.lambda_cross_modal is not None
        else model_defaults["lambda_cross_modal"]
    )  # <- 신규

    print(f"[{args.model}] 사용 하이퍼파라미터:")
    print(f"  lr={lr}, batch_size={batch_size}, patience={patience}, min_delta={min_delta}")
    print(f"  warmup_epochs={warmup_epochs}, eta_min={eta_min}")
    if args.model == "a2":
        print(f"  lambda_recon={lambda_recon}, lambda_cross_modal={lambda_cross_modal}, "
              f"use_full_ecg_branch={use_full_ecg_branch}")
    print()

    if args.sanity_check:
        run_sanity_check(
            args.model, use_reconstruction=use_reconstruction,
            use_full_ecg_branch=use_full_ecg_branch,   # <- sanity check에도 전달 필요
        )
        return

    train(
        args.model, args.epochs, batch_size, lr, args.num_workers,
        args.temperature, lambda_recon, use_reconstruction,
        args.use_cooccurrence_similarity, patience, min_delta,
        eta_min, warmup_epochs,
        lambda_cross_modal, use_full_ecg_branch,   # <- 신규 2개 전달
    )


if __name__ == "__main__":
    main()
