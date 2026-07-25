"""
Phase 1 - Step 2b: MELP(Wang et al., ICML 2025) 공개 checkpoint에서
frozen ECG embedding을 추출한다.

사전 준비:
    1) git clone https://github.com/HKU-MedAI/MELP.git
    2) cd MELP && pip install -r requirements.txt && pip install -e .
    3) https://huggingface.co/fuyingw/MELP_Encoder 에서 lightning .ckpt 다운로드
    4) config.py에서 MELP_REPO_ROOT, MELP_CKPT 경로 지정
    5) 아래가 에러 없이 되는지 먼저 확인:
           python -c "from melp.models.melp_model import MELPModel"
       (MELP는 MERL보다 의존성이 무겁습니다: timm, transformers, ot(POT), ipdb 등)

실행:
    python -m phase1_leakage_audit.extract_embeddings_melp --split train
    python -m phase1_leakage_audit.extract_embeddings_melp --split val
    python -m phase1_leakage_audit.extract_embeddings_melp --split test

출력:
    outputs/melp_unified_{split}.npz   (extract_embeddings_merl.py와 동일한 스키마)

주의:
    MELP 공개 코드(src/melp/datasets/finetune_dataset.py)는 lead index 4,5(aVL/aVF)를
    swap한다 (MIMIC-IV-ECG pretraining 시 사용한 lead 순서와 downstream 데이터셋의
    lead 순서가 달라서 생기는 보정). 그래서 이 스크립트는 lead_swap=True로 고정한다.
    MERL 스크립트와 값이 다른 것이 정상이니 임의로 통일하지 말 것.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.ptbxl_dataset import make_unified_dataset


def load_melp_encoder(ckpt_path):
    from melp.models.melp_model import MELPModel  # pip install -e . 필요 (MELP repo)

    model = MELPModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract(model, loader, device):
    z_list, form_list, rhythm_list, super_list, id_list = [], [], [], [], []
    for batch in tqdm(loader, desc="extracting MELP embeddings"):
        ecg = batch["ecg"].to(device)
        z = model.ext_ecg_emb(ecg)  # (B, 256), MELP 공식 API (models/melp_model.py 참고)
        z_list.append(z.cpu().numpy())
        form_list.append(batch["form_label"].numpy())
        rhythm_list.append(batch["rhythm_label"].numpy())
        super_list.append(batch["super_class_label"].numpy())
        id_list.extend(batch["ecg_id"].tolist() if torch.is_tensor(batch["ecg_id"]) else list(batch["ecg_id"]))

    return (
        np.concatenate(z_list, axis=0),
        np.concatenate(form_list, axis=0),
        np.concatenate(rhythm_list, axis=0),
        np.concatenate(super_list, axis=0),
        np.array(id_list),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = load_melp_encoder(config.MELP_CKPT).to(device)

    dataset = make_unified_dataset(
        split=args.split,
        ptbxl_root=config.PTBXL_ROOT,
        split_dir=config.SPLIT_DIR,
        lead_swap=True,  # MELP 자체 전처리 기준 (finetune_dataset.py의 ecg[[4,5]] swap)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    z, form_labels, rhythm_labels, super_labels, ecg_ids = extract(model, loader, device)

    label_cols = dataset.label_col_groups
    out_path = config.OUTPUT_DIR / f"melp_unified_{args.split}.npz"
    np.savez(
        out_path,
        z=z,
        form_labels=form_labels,
        rhythm_labels=rhythm_labels,
        super_labels=super_labels,
        form_names=np.array(label_cols["form"]),
        rhythm_names=np.array(label_cols["rhythm"]),
        super_names=np.array(label_cols["super_class"]),
        ecg_id=ecg_ids,
    )
    print(f"saved -> {out_path}  z.shape={z.shape}")


if __name__ == "__main__":
    main()
