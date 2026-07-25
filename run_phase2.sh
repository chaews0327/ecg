#!/usr/bin/env bash
# Phase 2 (A0 -> A1 -> A2 순차 구현 및 비교) 전체 실행.
# Phase 1(config.py, data/label_parser.py 실행 완료)이 선행되어 있어야 함.
#
# 기본은 A0/A1/A2 전체를 처음부터 재현. 특정 모델 아키텍처만 고쳐서 그
# 모델만 다시 돌리고 싶으면(예: A2 form encoder 수정 후 A2만 재학습) MODELS를
# 좁혀서 실행:
#   MODELS=a2 ./run_phase2.sh
# 이 경우에도 task 성능 요약/leakage audit은 항상 A0/A1/A2 전부를 비교한다
# (다른 모델 npz는 디스크에 이미 있는 걸 그대로 읽으므로 재학습 불필요).
#
# 학습을 건너뛰고 기존 checkpoint로 embedding 추출/평가만 다시 하고 싶으면:
#   SKIP_TRAINING=1 ./run_phase2.sh
#   SKIP_TRAINING=1 MODELS=a2 ./run_phase2.sh
#
# A2 전용 옵션(reconstruction, co-occurrence semantic similarity, cross-modal)은
# 환경변수로 조절:
#   LAMBDA_RECON=0.5 USE_COOCCURRENCE=1 ./run_phase2.sh   (기본값과 동일)
#   USE_COOCCURRENCE=0 ./run_phase2.sh                     (co-occurrence 끄고 overlap만)
#   NO_RECONSTRUCTION=1 MODELS=a2 ./run_phase2.sh          (A3 ablation: A2에서 recon만 제거)
#   LAMBDA_CROSS_MODAL=0.2 ./run_phase2.sh                 (cross-modal 가중치 조정)
#   NO_FULL_ECG_BRANCH=1 MODELS=a2 ./run_phase2.sh         (cross-modal branch 통째로 제거)
#
# 모델별 하이퍼파라미터 조정:
#   LR_A2=5e-5 BATCH_SIZE_A2=64 PATIENCE_A2=20 ./run_phase2.sh
#   LR_A0=3e-4 LR_A1=3e-4 ./run_phase2.sh
#   WARMUP_EPOCHS_A2=15 ETA_MIN_A2=1e-8 ./run_phase2.sh
set -e

export CUDA_VISIBLE_DEVICES=1
MODELS=${MODELS:-"a0 a1 a2"}
SKIP_TRAINING=${SKIP_TRAINING:-0}

LAMBDA_RECON=${LAMBDA_RECON:-0.5}
USE_COOCCURRENCE=${USE_COOCCURRENCE:-1}
NO_RECONSTRUCTION=${NO_RECONSTRUCTION:-0}
LAMBDA_CROSS_MODAL=${LAMBDA_CROSS_MODAL:-0.3}
NO_FULL_ECG_BRANCH=${NO_FULL_ECG_BRANCH:-0}

# --- 모델별 하이퍼파라미터 기본값 (환경변수로 override 가능) ---
# A0
LR_A0=${LR_A0:-2e-4}
BATCH_SIZE_A0=${BATCH_SIZE_A0:-256}
PATIENCE_A0=${PATIENCE_A0:-10}
MIN_DELTA_A0=${MIN_DELTA_A0:-1e-4}
WARMUP_EPOCHS_A0=${WARMUP_EPOCHS_A0:-5}
ETA_MIN_A0=${ETA_MIN_A0:-1e-6}

# A1
LR_A1=${LR_A1:-2e-4}
BATCH_SIZE_A1=${BATCH_SIZE_A1:-256}
PATIENCE_A1=${PATIENCE_A1:-10}
MIN_DELTA_A1=${MIN_DELTA_A1:-1e-4}
WARMUP_EPOCHS_A1=${WARMUP_EPOCHS_A1:-5}
ETA_MIN_A1=${ETA_MIN_A1:-1e-6}

# A2
LR_A2=${LR_A2:-1e-4}
BATCH_SIZE_A2=${BATCH_SIZE_A2:-128}
PATIENCE_A2=${PATIENCE_A2:-15}
MIN_DELTA_A2=${MIN_DELTA_A2:-5e-5}
WARMUP_EPOCHS_A2=${WARMUP_EPOCHS_A2:-10}
ETA_MIN_A2=${ETA_MIN_A2:-1e-7}

echo "=== [0/5] Phase 2 로직 sanity check (실제 데이터 불필요) ==="
python -m tests.sanity_check_phase2

echo ""
echo "=== [0.5/5] torch 모델 배선 확인 (실제 GPU/torch 필요, 각각 몇 초면 끝남) ==="
if [ "$SKIP_TRAINING" == "1" ]; then
  echo "SKIP_TRAINING=1 이므로 건너뜀"
else
  for m in $MODELS; do
    extra_args=""
    if [ "$m" == "a2" ]; then
      if [ "$NO_RECONSTRUCTION" == "1" ]; then
        extra_args="$extra_args --no_reconstruction"
      fi
      if [ "$NO_FULL_ECG_BRANCH" == "1" ]; then
        extra_args="$extra_args --no_full_ecg_branch"
      fi
    fi
    python -m phase2_model_training.train_phase2 --model $m --sanity_check $extra_args
  done
fi

echo ""
echo "=== [1/5] 학습 (MODELS=\"$MODELS\") ==="
if [ "$SKIP_TRAINING" == "1" ]; then
  echo "SKIP_TRAINING=1 이므로 학습 건너뜀 (기존 checkpoint를 그대로 사용)"
else
  echo "하이퍼파라미터 설정:"
  echo "  A0: lr=$LR_A0, batch_size=$BATCH_SIZE_A0, patience=$PATIENCE_A0, min_delta=$MIN_DELTA_A0, warmup_epochs=$WARMUP_EPOCHS_A0, eta_min=$ETA_MIN_A0"
  echo "  A1: lr=$LR_A1, batch_size=$BATCH_SIZE_A1, patience=$PATIENCE_A1, min_delta=$MIN_DELTA_A1, warmup_epochs=$WARMUP_EPOCHS_A1, eta_min=$ETA_MIN_A1"
  echo "  A2: lr=$LR_A2, batch_size=$BATCH_SIZE_A2, patience=$PATIENCE_A2, min_delta=$MIN_DELTA_A2, warmup_epochs=$WARMUP_EPOCHS_A2, eta_min=$ETA_MIN_A2, lambda_recon=$LAMBDA_RECON, lambda_cross_modal=$LAMBDA_CROSS_MODAL"
  echo "  A2 추가: USE_COOCCURRENCE=$USE_COOCCURRENCE, NO_RECONSTRUCTION=$NO_RECONSTRUCTION, NO_FULL_ECG_BRANCH=$NO_FULL_ECG_BRANCH"
  echo ""

  declare -A MODEL_LR
  declare -A MODEL_BATCH_SIZE
  declare -A MODEL_PATIENCE
  declare -A MODEL_MIN_DELTA
  declare -A MODEL_WARMUP_EPOCHS
  declare -A MODEL_ETA_MIN

  MODEL_LR[a0]=$LR_A0
  MODEL_BATCH_SIZE[a0]=$BATCH_SIZE_A0
  MODEL_PATIENCE[a0]=$PATIENCE_A0
  MODEL_MIN_DELTA[a0]=$MIN_DELTA_A0
  MODEL_WARMUP_EPOCHS[a0]=$WARMUP_EPOCHS_A0
  MODEL_ETA_MIN[a0]=$ETA_MIN_A0

  MODEL_LR[a1]=$LR_A1
  MODEL_BATCH_SIZE[a1]=$BATCH_SIZE_A1
  MODEL_PATIENCE[a1]=$PATIENCE_A1
  MODEL_MIN_DELTA[a1]=$MIN_DELTA_A1
  MODEL_WARMUP_EPOCHS[a1]=$WARMUP_EPOCHS_A1
  MODEL_ETA_MIN[a1]=$ETA_MIN_A1

  MODEL_LR[a2]=$LR_A2
  MODEL_BATCH_SIZE[a2]=$BATCH_SIZE_A2
  MODEL_PATIENCE[a2]=$PATIENCE_A2
  MODEL_MIN_DELTA[a2]=$MIN_DELTA_A2
  MODEL_WARMUP_EPOCHS[a2]=$WARMUP_EPOCHS_A2
  MODEL_ETA_MIN[a2]=$ETA_MIN_A2

  for m in $MODELS; do
    echo "--- $m ---"

    # 기본 하이퍼파라미터 (A0/A1/A2 모두) - cosine warmup 옵션 포함
    extra_args="--lr ${MODEL_LR[$m]} --batch_size ${MODEL_BATCH_SIZE[$m]} "
    extra_args="$extra_args --patience ${MODEL_PATIENCE[$m]} --min_delta ${MODEL_MIN_DELTA[$m]}"
    extra_args="$extra_args --warmup_epochs ${MODEL_WARMUP_EPOCHS[$m]} --eta_min ${MODEL_ETA_MIN[$m]}"

    # A2 전용: reconstruction / cross-modal / co-occurrence 관련 옵션
    if [ "$m" == "a2" ]; then
      extra_args="$extra_args --lambda_recon $LAMBDA_RECON"
      extra_args="$extra_args --lambda_cross_modal $LAMBDA_CROSS_MODAL"
      if [ "$NO_FULL_ECG_BRANCH" == "1" ]; then
        extra_args="$extra_args --no_full_ecg_branch"
      fi
      if [ "$USE_COOCCURRENCE" == "1" ]; then
        extra_args="$extra_args --use_cooccurrence_similarity"
      fi
      if [ "$NO_RECONSTRUCTION" == "1" ]; then
        extra_args="$extra_args --no_reconstruction"
      fi
    fi

    echo "    (옵션: $extra_args)"
    python -m phase2_model_training.train_phase2 --model $m --epochs 100 $extra_args
  done
fi

echo ""
echo "=== [2/5] Embedding 추출 (MODELS=\"$MODELS\", train/val/test) ==="
for m in $MODELS; do
  for split in train val test; do
    python -m phase2_model_training.extract_embeddings_phase2 --model $m --split $split
  done
done

echo ""
echo "=== [3/5] 기본 task 성능 (A0/A1/A2) ==="
python -m phase2_model_training.print_task_auc_summary

echo ""
echo "=== [4/5] Incremental leakage 비교 (핵심 결과, A1 vs A2, z_form/z_rhythm 각각) ==="
python -m phase2_model_training.audit_incremental_leakage

echo ""
echo "완료. outputs/phase2_incremental_leakage_summary.json 을 확인하세요."
echo "이번 실행 설정:"
echo "  SKIP_TRAINING=$SKIP_TRAINING"
echo "  LAMBDA_RECON=$LAMBDA_RECON  USE_COOCCURRENCE=$USE_COOCCURRENCE  NO_RECONSTRUCTION=$NO_RECONSTRUCTION"
echo "  LAMBDA_CROSS_MODAL=$LAMBDA_CROSS_MODAL  NO_FULL_ECG_BRANCH=$NO_FULL_ECG_BRANCH"
echo "  LR: A0=$LR_A0, A1=$LR_A1, A2=$LR_A2"
echo "  BATCH_SIZE: A0=$BATCH_SIZE_A0, A1=$BATCH_SIZE_A1, A2=$BATCH_SIZE_A2"
echo "  PATIENCE: A0=$PATIENCE_A0, A1=$PATIENCE_A1, A2=$PATIENCE_A2"
echo "  WARMUP_EPOCHS: A0=$WARMUP_EPOCHS_A0, A1=$WARMUP_EPOCHS_A1, A2=$WARMUP_EPOCHS_A2"
echo "  ETA_MIN: A0=$ETA_MIN_A0, A1=$ETA_MIN_A1, A2=$ETA_MIN_A2"