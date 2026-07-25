#!/usr/bin/env bash
# Phase 1(문제 검증: leakage audit) 전체를 순서대로 실행한다.
# 실행 전 config.py의 경로들을 반드시 먼저 설정할 것.
set -e

export CUDA_VISIBLE_DEVICES=0

echo "=== [0/4] 파이프라인 배선 sanity check (합성 데이터, 실제 PTB-XL 불필요) ==="
python -m tests.sanity_check

echo ""
echo "=== [1/4] PTB-XL 라벨/스플릿 생성 ==="
python -m data.label_parser

echo ""
echo "=== [2/4] MERL embedding 추출 (train/val/test) ==="
python -m phase1_leakage_audit.extract_embeddings_merl --split train
python -m phase1_leakage_audit.extract_embeddings_merl --split val
python -m phase1_leakage_audit.extract_embeddings_merl --split test

echo ""
echo "=== [3/4] MELP embedding 추출 (train/val/test) ==="
python -m phase1_leakage_audit.extract_embeddings_melp --split train
python -m phase1_leakage_audit.extract_embeddings_melp --split val
python -m phase1_leakage_audit.extract_embeddings_melp --split test

echo ""
echo "=== [4/4] Leakage audit 실행 ==="
python -m phase1_leakage_audit.audit_basic
python -m phase1_leakage_audit.audit_confound_control

echo ""
echo "완료. outputs/phase1_leakage_summary.json 을 확인하세요."
