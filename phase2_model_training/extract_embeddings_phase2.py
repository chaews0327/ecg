"""
학습된 A0/A1/A2 checkpoint에서 frozen embedding을 추출한다.
phase1_leakage_audit/extract_embeddings_merl.py와 동일한 패턴(모델 로드 ->
frozen forward -> npz 저장)을 그대로 따른다.

실행:
    python -m phase2_model_training.extract_embeddings_phase2 --model a0 --split train
    python -m phase2_model_training.extract_embeddings_phase2 --model a1 --split test
    python -m phase2_model_training.extract_embeddings_phase2 --model a2 --split test

출력:
    outputs/phase2_{model}_{split}.npz
        (a0)     z (N, 512)
        (a1, a2) z_form (N, 256), z_rhythm (N, 256)
        공통: form_labels, rhythm_labels, super_labels, *_names, ecg_id
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.ptbxl_dataset import make_unified_dataset
from phase2_model_training.models.a0_monolithic import A0Monolithic
from phase2_model_training.models.a1_dual_raw import A1DualRaw
from phase2_model_training.models.a2_grain_matched import A2GrainMatched
from phase2_model_training.preprocessing.phase2_dataset import make_grain_matched_dataset


MODEL_REGISTRY = {"a0": A0Monolithic, "a1": A1DualRaw, "a2": A2GrainMatched}


@torch.no_grad()
def extract(model_name: str, model, loader, device):
    z_keys = ["z"] if model_name == "a0" else ["z_form", "z_rhythm"]
    collected = {k: [] for k in z_keys}
    form_list, rhythm_list, super_list, id_list = [], [], [], []

    for batch in tqdm(loader, desc=f"extracting {model_name} embeddings"):
        batch_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_device)
        for k in z_keys:
            collected[k].append(out[k].cpu().numpy())
        form_list.append(batch["form_label"].numpy())
        rhythm_list.append(batch["rhythm_label"].numpy())
        super_list.append(batch["super_class_label"].numpy())
        id_list.extend(batch["ecg_id"].tolist() if torch.is_tensor(batch["ecg_id"]) else list(batch["ecg_id"]))

    result = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    result["form_labels"] = np.concatenate(form_list, axis=0)
    result["rhythm_labels"] = np.concatenate(rhythm_list, axis=0)
    result["super_labels"] = np.concatenate(super_list, axis=0)
    result["ecg_id"] = np.array(id_list)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=None, help="기본값: outputs/phase2_checkpoints/{model}_best.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = MODEL_REGISTRY[args.model]().to(device)
    ckpt_path = args.ckpt or str(config.OUTPUT_DIR / "phase2_checkpoints" / f"{args.model}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"checkpoint 로드 완료: {ckpt_path}")

    if args.model in ("a0", "a1"):
        dataset = make_unified_dataset(split=args.split, ptbxl_root=config.PTBXL_ROOT, split_dir=config.SPLIT_DIR)
    else:
        dataset = make_grain_matched_dataset(split=args.split, ptbxl_root=config.PTBXL_ROOT, split_dir=config.SPLIT_DIR)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    result = extract(args.model, model, loader, device)

    label_cols = dataset.label_col_groups
    result["form_names"] = np.array(label_cols["form"])
    result["rhythm_names"] = np.array(label_cols["rhythm"])
    result["super_names"] = np.array(label_cols["super_class"])

    out_path = config.OUTPUT_DIR / f"phase2_{args.model}_{args.split}.npz"
    np.savez(out_path, **result)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
