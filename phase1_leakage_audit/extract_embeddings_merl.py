"""
Phase 1 - Step 2a: MERL(Liu et al., ICML 2024) 공개 checkpoint에서
frozen ECG embedding을 추출한다.

사전 준비:
    1) git clone https://github.com/cheliu-computation/MERL-ICML2024.git
    2) MERL README의 Google Drive 링크에서 "xxx_encoder.pth"(ResNet18) 다운로드
       (xxx_ckpt.pth 전체가 아니라 "encoder"라고 적힌 쪽 -- ECG encoder만 있으면 됨)
    3) config.py에서 MERL_REPO_ROOT, MERL_ENCODER_CKPT 경로 지정
    4) python -m data.label_parser 를 먼저 실행해 data_split/ 이하 csv를 만들어 둘 것

실행:
    python -m phase1_leakage_audit.extract_embeddings_merl --split train
    python -m phase1_leakage_audit.extract_embeddings_merl --split val
    python -m phase1_leakage_audit.extract_embeddings_merl --split test

출력:
    outputs/merl_unified_{split}.npz
        z              (N, 512)  frozen ECG embedding
        form_labels    (N, 19)
        rhythm_labels  (N, 12)
        super_labels   (N, 5)
        form_names, rhythm_names, super_names
        ecg_id         (N,)

주의:
    MERL의 ResNet18은 12-lead raw 입력을 그대로 받고, lead 순서를 바꾸지 않는다
    (MERL 자체 finetune_dataset.py 기준). 따라서 lead_swap=False로 고정한다.
    이 swap 여부는 checkpoint마다 다르므로 MELP용 스크립트와 값이 다른 것이 정상이다.
"""
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.ptbxl_dataset import make_unified_dataset


def load_merl_encoder(ckpt_path, num_leads: int = 12):
    """MERL repo의 ResNet18을 불러와, 최종 linear 분류층을 제거하고
    512차원 pooled feature만 뽑도록 wrapping한다.
    
    주의: checkpoint의 linear layer가 임의의 클래스 수(예: 10)로 저장되어 있을 수 있으므로,
    state_dict 로드 전에 linear.weight/linear.bias를 먼저 제거한다.
    """
    sys.path.insert(0, str(config.MERL_REPO_ROOT / "finetune"))
    from models.resnet1d import ResNet18  # noqa: MERL repo 내부 모듈, 위 sys.path 삽입 후에만 import 가능

    model = ResNet18(num_classes=10)  # checkpoint의 실제 클래스 수에 맞춤 (일단 일반적인 10을 기본값으로)
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # linear layer는 어차피 버릴 것이므로, state_dict에서 미리 제거해 차원 불일치를 피한다
    state_dict_filtered = {k: v for k, v in state_dict.items() 
                           if not k.startswith("linear.")}
    
    missing, unexpected = model.load_state_dict(state_dict_filtered, strict=False)
    print(f"[MERL] state_dict 로딩 완료. missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 2:  # 보통 linear.weight/linear.bias 2개만 missing이어야 정상
        print(f"[MERL][경고] missing keys가 예상(2개)보다 많습니다: {missing}")

    model.linear = torch.nn.Identity()  # (B, 512) pooled feature만 출력하도록 교체
    model.eval()
    return model


@torch.no_grad()
def extract(model, loader, device):
    z_list, form_list, rhythm_list, super_list, id_list = [], [], [], [], []
    for batch in tqdm(loader, desc="extracting MERL embeddings"):
        ecg = batch["ecg"].to(device)
        z = model(ecg)  # (B, 512)
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

    model = load_merl_encoder(config.MERL_ENCODER_CKPT).to(device)

    dataset = make_unified_dataset(
        split=args.split,
        ptbxl_root=config.PTBXL_ROOT,
        split_dir=config.SPLIT_DIR,
        lead_swap=False,  # MERL 자체 전처리 기준
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    z, form_labels, rhythm_labels, super_labels, ecg_ids = extract(model, loader, device)

    label_cols = dataset.label_col_groups
    out_path = config.OUTPUT_DIR / f"merl_unified_{args.split}.npz"
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
