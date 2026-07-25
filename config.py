"""
공통 경로/상수 설정.

이 파일 하나만 본인 환경에 맞게 수정하면, 나머지 모든 스크립트는
여기서 정의한 경로를 그대로 import해서 씁니다.
"""
from pathlib import Path

# ============================================================
# 필수: 아래 경로들을 실제 환경에 맞게 수정하세요.
# ============================================================

# PTB-XL 원본 데이터 루트.
# 이 폴더 안에 ptbxl_database.csv, scp_statements.csv, records500/ 가 있어야 합니다.
# 다운로드: https://physionet.org/content/ptb-xl/1.0.3/
PTBXL_ROOT = Path("/data/ecg/public/ptb-xl")

# label_parser.py가 생성하는 split csv를 저장할 위치
SPLIT_DIR = Path("./data_split")

# ---- MERL (Liu et al., ICML 2024) ----
# git clone https://github.com/cheliu-computation/MERL-ICML2024.git
MERL_REPO_ROOT = Path("/home/chaewon/medicalai/new/MERL-ICML2024")
# MERL README의 Google Drive 링크에서 다운로드하는 "xxx_encoder.pth" (ResNet18 backbone only)
MERL_ENCODER_CKPT = Path("/home/chaewon/medicalai/new/MERL-ICML2024/res18_best_encoder.pth")

# ---- MELP (Wang et al., ICML 2025) ----
# git clone https://github.com/HKU-MedAI/MELP.git
# cd MELP && pip install -r requirements.txt && pip install -e .
MELP_REPO_ROOT = Path("/home/chaewon/medicalai/new/MELP")
# https://huggingface.co/fuyingw/MELP_Encoder 에서 다운로드하는 lightning .ckpt 파일
MELP_CKPT = Path("/home/chaewon/medicalai/MELP/scripts/logs/melp/ckpts/melp_melp_2026_03_27_15_30_34/epoch=3-step=46592.ckpt")

# ============================================================
# 아래는 보통 수정할 필요 없음
# ============================================================

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SPLIT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42

# PTB-XL 공식 벤치마크 protocol (Strodthoff et al., 2020):
# fold 1-8 = train, fold 9 = validation, fold 10 = test
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# PTB-XL diagnostic superclass 5개 (고정)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
