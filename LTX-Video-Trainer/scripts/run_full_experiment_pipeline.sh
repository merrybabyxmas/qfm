#!/usr/bin/env bash
# =============================================================================
# QSFM Full Experiment Pipeline  (논문 전체 실험 자동화)
# =============================================================================
#
#  PHASE 0: 환경 점검 (conda 자동 감지)
#  PHASE 1: 데이터셋 준비
#           1a. PISSA (fxmeng/pissa-dataset, HuggingFace) ← 기본
#           1b. Moving MNIST  (Tier 1)
#           1c. UCF-101       (Tier 2, 합성 모드)
#  PHASE 2: LTX-Video VAE 전처리
#  PHASE 3: QSFM 학습
#  PHASE 4: Baseline 추론 (Auto-regressive, FreeNoise, LTX-V Pure)
#  PHASE 5: Table 2: Scaling Law Benchmark
#  PHASE 6: 평가 워크스페이스 업데이트
#  PHASE 7: Table 1: 통합 평가
#  PHASE 8: 결과 요약 출력
#
# 사용법 (conda activate 없이도 동작):
#   cd /home/dongwoo43/qfm/LTX-Video-Trainer
#   bash scripts/run_full_experiment_pipeline.sh [OPTIONS]
#
# 옵션:
#   --skip-training      학습 단계 스킵
#   --skip-baseline      Baseline 추론 스킵
#   --skip-pissa         PISSA 데이터셋 준비 스킵
#   --skip-mnist         Moving MNIST 단계 스킵
#   --skip-ucf           UCF-101 단계 스킵
#   --quick              빠른 테스트 (steps=200)
#   --steps N            학습 스텝 수 (기본 1000)
#   --pissa-samples N    PISSA 샘플 수 (기본 40)
# =============================================================================

set -euo pipefail

# ── 색상 코드 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()     { echo -e "${GREEN}✅${NC} $*"; }
warn()   { echo -e "${YELLOW}⚠️ ${NC} $*"; }
err()    { echo -e "${RED}❌${NC} $*"; }
phase()  { echo -e "\n${BOLD}${BLUE}════════════════════════════════════════${NC}"; \
           echo -e "${BOLD}${BLUE} $* ${NC}"; \
           echo -e "${BOLD}${BLUE}════════════════════════════════════════${NC}\n"; }

# ── conda 환경 자동 감지 ──────────────────────────────────────────────────────
CONDA_ENV_NAME="afm"
CONDA_PYTHON=""

# 1순위: 이미 활성화된 환경
if python -c "import torch" 2>/dev/null; then
    CONDA_PYTHON="$(which python)"
    ok "현재 환경에서 torch 확인: $CONDA_PYTHON"
# 2순위: conda run
elif command -v conda &>/dev/null; then
    CONDA_PYTHON="$(conda run -n $CONDA_ENV_NAME which python 2>/dev/null || true)"
    if [ -n "$CONDA_PYTHON" ]; then
        ok "conda 환경 감지: $CONDA_PYTHON"
        # conda 환경의 PATH를 현재 세션에 반영
        CONDA_BASE="$(conda info --base 2>/dev/null || echo '')"
        if [ -n "$CONDA_BASE" ]; then
            export PATH="$CONDA_BASE/envs/$CONDA_ENV_NAME/bin:$PATH"
        fi
    fi
# 3순위: 직접 경로
elif [ -f "/home/dongwoo43/miniconda3/envs/$CONDA_ENV_NAME/bin/python" ]; then
    CONDA_PYTHON="/home/dongwoo43/miniconda3/envs/$CONDA_ENV_NAME/bin/python"
    export PATH="/home/dongwoo43/miniconda3/envs/$CONDA_ENV_NAME/bin:$PATH"
    ok "conda 환경 직접 경로 사용: $CONDA_PYTHON"
else
    err "conda 환경 '$CONDA_ENV_NAME' 을 찾을 수 없습니다."
    err "실행 전 'conda activate $CONDA_ENV_NAME' 을 먼저 실행하거나,"
    err "conda가 설치되어 있는지 확인하세요."
    exit 1
fi

# python 명령이 올바른지 최종 확인
python -c "import torch" 2>/dev/null || {
    err "torch import 실패. conda 환경을 확인하세요: conda activate $CONDA_ENV_NAME"
    exit 1
}

# ── 옵션 파싱 ────────────────────────────────────────────────────────────────
SKIP_TRAINING=false
SKIP_BASELINE=false
SKIP_PISSA=false
SKIP_MNIST=false
SKIP_UCF=false
TRAIN_STEPS=1000
QUICK=false
PISSA_SAMPLES=40

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case $arg in
        --skip-training) SKIP_TRAINING=true ;;
        --skip-baseline) SKIP_BASELINE=true ;;
        --skip-pissa)    SKIP_PISSA=true ;;
        --skip-mnist)    SKIP_MNIST=true ;;
        --skip-ucf)      SKIP_UCF=true ;;
        --quick)         QUICK=true; TRAIN_STEPS=200 ;;
        --steps)
            i=$((i+1)); TRAIN_STEPS="${!i}" ;;
        --pissa-samples)
            i=$((i+1)); PISSA_SAMPLES="${!i}" ;;
    esac
    i=$((i+1))
done

if $QUICK; then
    PISSA_SAMPLES=16
    warn "QUICK 모드: steps=${TRAIN_STEPS}, pissa_samples=${PISSA_SAMPLES}"
fi

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE=/home/dongwoo43/qfm
TRAINER=$BASE/LTX-Video-Trainer
DATA_BBB=$BASE/qsfm_data
DATA_PISSA=$BASE/pissa_data
DATA_MNIST=$BASE/mnist_data
DATA_UCF=$BASE/ucf101_data
WORKSPACE=$BASE/eval_workspace
RESULTS=$WORKSPACE/eval_results

cd $TRAINER
export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── 시작 배너 ────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
cat << 'EOF'
  ██████╗ ███████╗███████╗███╗   ███╗
 ██╔═══██╗██╔════╝██╔════╝████╗ ████║
 ██║   ██║███████╗█████╗  ██╔████╔██║
 ██║▄▄ ██║╚════██║██╔══╝  ██║╚██╔╝██║
 ╚██████╔╝███████║██║     ██║ ╚═╝ ██║
  ╚══▀▀═╝ ╚══════╝╚═╝     ╚═╝     ╚═╝
  Quantum Superposition Flow Matching
  Full Experiment Pipeline
EOF
echo -e "${NC}"
echo "  시작: $(date)"
echo "  경로: $TRAINER"
echo "  Steps: $TRAIN_STEPS"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
phase "PHASE 0: 환경 점검"
# ─────────────────────────────────────────────────────────────────────────────

log "Python 환경 확인..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torchquantum; print(f'TorchQuantum OK')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')" 2>/dev/null || echo "N/A")
log "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo 'CPU only')  (VRAM: $GPU_MEM)"
ok "환경 점검 완료"

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_PISSA; then
phase "PHASE 1a: PISSA 데이터셋 준비 (fxmeng/pissa-dataset, HuggingFace)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -f "$DATA_PISSA/dataset.json" ]; then
        N_PISSA=$(python -c "import json; print(len(json.load(open('$DATA_PISSA/dataset.json'))))")
        ok "PISSA 데이터셋: $N_PISSA 클립 이미 준비됨 (스킵)"
    else
        log "fxmeng/pissa-dataset 다운로드 및 합성 비디오 생성..."
        python scripts/prepare_pissa_dataset.py \
            --output_dir $DATA_PISSA \
            --n_samples $PISSA_SAMPLES
        ok "PISSA 데이터셋 준비 완료 ($PISSA_SAMPLES 샘플)"
    fi
else
    warn "PISSA 데이터셋 스킵 (--skip-pissa)"
fi

# Big Buck Bunny 데이터 존재 여부만 확인 (fallback)
if [ -f "$DATA_BBB/dataset.json" ]; then
    N_BBB=$(python -c "import json; print(len(json.load(open('$DATA_BBB/dataset.json'))))")
    ok "Big Buck Bunny 보조 데이터: $N_BBB 씬"
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_MNIST; then
phase "PHASE 1b: Moving MNIST 준비 (Tier 1)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -f "$DATA_MNIST/dataset.json" ]; then
        N_MNIST=$(python -c "import json; print(len(json.load(open('$DATA_MNIST/dataset.json'))))")
        ok "Moving MNIST: $N_MNIST 시퀀스 이미 준비됨 (스킵)"
    else
        log "Moving MNIST 다운로드 및 준비..."
        python scripts/prepare_moving_mnist.py \
            --output_dir $DATA_MNIST \
            --n_samples 32
        ok "Moving MNIST 준비 완료"
    fi
else
    warn "Moving MNIST 스킵 (--skip-mnist)"
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_UCF; then
phase "PHASE 1c: UCF-101 준비 (Tier 2, 합성 모드)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -f "$DATA_UCF/dataset.json" ]; then
        N_UCF=$(python -c "import json; print(len(json.load(open('$DATA_UCF/dataset.json'))))")
        ok "UCF-101: $N_UCF 클립 이미 준비됨 (스킵)"
    else
        log "UCF-101 준비 (합성 동작 비디오)..."
        python scripts/prepare_ucf101.py \
            --output_dir $DATA_UCF \
            --n_classes 5 \
            --clips_per_class 8 \
            --synthetic_only
        ok "UCF-101 (합성) 준비 완료"
    fi
else
    warn "UCF-101 스킵 (--skip-ucf)"
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_PISSA && [ -f "$DATA_PISSA/dataset.json" ]; then
phase "PHASE 2a: PISSA VAE 전처리"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$DATA_PISSA/latents" ]; then
        ok "PISSA latents 이미 존재 (스킵)"
    else
        log "PISSA 데이터셋 VAE 인코딩..."
        python scripts/preprocess_dataset.py \
            $DATA_PISSA/dataset.json \
            --output-dir $DATA_PISSA \
            --resolution-buckets "320x240x48" \
            --device cuda
        ok "PISSA 전처리 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_MNIST && [ -f "$DATA_MNIST/dataset.json" ]; then
phase "PHASE 2b: Moving MNIST VAE 전처리"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$DATA_MNIST/latents" ]; then
        ok "Moving MNIST latents 이미 존재 (스킵)"
    else
        log "Moving MNIST VAE 인코딩..."
        python scripts/preprocess_dataset.py \
            $DATA_MNIST/dataset.json \
            --output-dir $DATA_MNIST \
            --resolution-buckets "64x64x20" \
            --device cuda
        ok "Moving MNIST 전처리 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_UCF && [ -f "$DATA_UCF/dataset.json" ]; then
phase "PHASE 2c: UCF-101 VAE 전처리"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$DATA_UCF/latents" ]; then
        ok "UCF-101 latents 이미 존재 (스킵)"
    else
        log "UCF-101 VAE 인코딩..."
        python scripts/preprocess_dataset.py \
            $DATA_UCF/dataset.json \
            --output-dir $DATA_UCF \
            --resolution-buckets "320x240x48" \
            --device cuda
        ok "UCF-101 전처리 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_TRAINING; then
phase "PHASE 3a: QSFM 학습 — PISSA 데이터셋 (K=4)"
# ─────────────────────────────────────────────────────────────────────────────
    # PISSA latents 우선, 없으면 Big Buck Bunny fallback
    if [ -d "$DATA_PISSA/latents" ]; then
        TRAIN_DATA=$DATA_PISSA
        log "학습 데이터: PISSA ($DATA_PISSA)"
    elif [ -d "$DATA_BBB/latents" ]; then
        TRAIN_DATA=$DATA_BBB
        warn "PISSA latents 없음 → Big Buck Bunny 데이터로 학습"
    else
        err "학습 데이터 없음. PISSA 전처리 먼저 실행 필요."
        err "  python scripts/prepare_pissa_dataset.py --output_dir $DATA_PISSA"
        err "  PYTHONPATH=src python scripts/preprocess_dataset.py ..."
        exit 1
    fi

    if [ -d "$TRAINER/outputs/qsfm_lora/checkpoints" ]; then
        ok "QSFM 체크포인트 이미 존재 (스킵)"
    else
        log "QSFM 학습 시작 (steps=$TRAIN_STEPS, data=$TRAIN_DATA)..."
        python -c "
import yaml
with open('configs/qsfm_2b_lora.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['preprocessed_data_root'] = '$TRAIN_DATA'
cfg['optimization']['steps'] = $TRAIN_STEPS
with open('/tmp/qsfm_pissa.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('config 생성: /tmp/qsfm_pissa.yaml')
"
        python scripts/train.py --config /tmp/qsfm_pissa.yaml
        ok "QSFM 학습 완료"
    fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_MNIST && [ -d "$DATA_MNIST/latents" ]; then
phase "PHASE 3b: QSFM 학습 — Moving MNIST (K=4)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "outputs/qsfm_mnist/checkpoints" ]; then
        ok "Moving MNIST QSFM 체크포인트 이미 존재"
    else
        log "Moving MNIST QSFM 학습..."
        # MNIST용 임시 config (데이터 경로만 변경)
        python -c "
import yaml, copy
with open('configs/qsfm_2b_lora.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['preprocessed_data_root'] = '$DATA_MNIST'
cfg['output_dir'] = 'outputs/qsfm_mnist'
cfg['optimization']['steps'] = $TRAIN_STEPS
with open('/tmp/qsfm_mnist.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
        python scripts/train.py --config /tmp/qsfm_mnist.yaml
        ok "Moving MNIST QSFM 학습 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_UCF && [ -d "$DATA_UCF/latents" ]; then
phase "PHASE 3c: QSFM 학습 — UCF-101 (K=4)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "outputs/qsfm_ucf/checkpoints" ]; then
        ok "UCF-101 QSFM 체크포인트 이미 존재"
    else
        log "UCF-101 QSFM 학습..."
        python -c "
import yaml
with open('configs/qsfm_2b_lora.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['preprocessed_data_root'] = '$DATA_UCF'
cfg['output_dir'] = 'outputs/qsfm_ucf'
cfg['optimization']['steps'] = $TRAIN_STEPS
with open('/tmp/qsfm_ucf.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
        python scripts/train.py --config /tmp/qsfm_ucf.yaml
        ok "UCF-101 QSFM 학습 완료"
    fi
fi
fi  # end if ! $SKIP_TRAINING

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_BASELINE; then
phase "PHASE 4: Baseline 추론 (LTX-Video Pure + FreeNoise + StreamingT2V + Open-Sora)"
# ─────────────────────────────────────────────────────────────────────────────
    PURE_DIR=$WORKSPACE/baselines/ltx_video_pure
    if ls $PURE_DIR/shot_*.mp4 2>/dev/null | grep -q "shot"; then
        ok "LTX-Video Pure 비디오 이미 존재 (스킵)"
    else
        log "LTX-Video Pure 추론 중 (QSFM 없음)..."
        python scripts/run_ltxv_pure_inference.py \
            --output_dir $PURE_DIR \
            --prompts_json $WORKSPACE/prompts.json \
            --steps 30
        ok "LTX-Video Pure 추론 완료"
    fi

    # ── FreeNoise ──────────────────────────────────────────────────────────
    FN_DIR=$WORKSPACE/baselines/free_noise
    if ls $FN_DIR/shot_*.mp4 2>/dev/null | grep -q "shot"; then
        ok "FreeNoise 비디오 이미 존재 (스킵)"
    else
        log "FreeNoise 추론 중 (Noise Rescheduling)..."
        python scripts/run_free_noise_inference.py \
            --output_dir $FN_DIR \
            --prompts_json $WORKSPACE/prompts.json \
            --steps 20 \
            --width 512 --height 320 --num_frames 49
        ok "FreeNoise 추론 완료"
    fi

    # ── StreamingT2V ───────────────────────────────────────────────────────
    ST2V_DIR=$WORKSPACE/baselines/streaming_t2v
    if ls $ST2V_DIR/shot_*.mp4 2>/dev/null | grep -q "shot"; then
        ok "StreamingT2V 비디오 이미 존재 (스킵)"
    else
        log "StreamingT2V 추론 중 (Sliding-Window 조건부 생성)..."
        python scripts/run_streaming_t2v_inference.py \
            --output_dir $ST2V_DIR \
            --prompts_json $WORKSPACE/prompts.json \
            --steps 20 \
            --width 512 --height 320 --num_frames 49 \
            --conditioning_strength 0.6
        ok "StreamingT2V 추론 완료"
    fi

    # ── Open-Sora (CogVideoX) ─────────────────────────────────────────────
    SORA_DIR=$WORKSPACE/baselines/open_sora
    if ls $SORA_DIR/shot_*.mp4 2>/dev/null | grep -q "shot"; then
        ok "Open-Sora(CogVideoX) 비디오 이미 존재 (스킵)"
    else
        log "Open-Sora(CogVideoX-2b) 추론 중 (~10GB 다운로드)..."
        python scripts/run_open_sora_inference.py \
            --output_dir $SORA_DIR \
            --prompts_json $WORKSPACE/prompts.json \
            --steps 50 \
            --width 480 --height 272 --num_frames 49
        ok "Open-Sora(CogVideoX) 추론 완료"
    fi
else
    warn "모든 Baseline 추론 스킵 (--skip-baseline)"
fi

# ─────────────────────────────────────────────────────────────────────────────
phase "PHASE 5: Scaling Law Benchmark (Table 2)"
# ─────────────────────────────────────────────────────────────────────────────

log "K=4,8,16,32 VRAM/Latency 측정..."
python scripts/benchmark_scaling.py \
    --n_runs 5 \
    --output_csv $RESULTS/table2_scaling.csv
ok "Scaling benchmark 완료 → $RESULTS/table2_scaling.csv"

# ─────────────────────────────────────────────────────────────────────────────
phase "PHASE 6: 평가 워크스페이스 업데이트"
# ─────────────────────────────────────────────────────────────────────────────

log "워크스페이스 구조 업데이트..."
python scripts/setup_eval_workspace.py \
    --workspace $WORKSPACE \
    --qsfm_output outputs/qsfm_lora
ok "워크스페이스 업데이트 완료"

# QSFM ablation 결과도 workspace에 복사
log "Ablation 결과 복사..."
for variant in qsfm_no_hamiltonian qsfm_gaussian; do
    src=$WORKSPACE/qsfm_outputs/$variant
    if ls $src/shot_*.mp4 2>/dev/null | grep -q "shot"; then
        ok "  $variant: 이미 배치됨"
    else
        warn "  $variant: 비디오 없음 (ablation 학습 후 배치 필요)"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
phase "PHASE 7: 통합 평가 — Table 1 (CLIPSIM + Consistency + FVD)"
# ─────────────────────────────────────────────────────────────────────────────

log "전체 모델 CLIPSIM + Temporal Consistency 계산..."
python scripts/run_master_eval.py \
    --workspace $WORKSPACE \
    --compute_fvd \
    --use_r3d \
    --output_csv $RESULTS/table1_full.csv

ok "Table 1 완료 → $RESULTS/table1_full.csv"

# ─────────────────────────────────────────────────────────────────────────────
phase "PHASE 8: 최종 결과 요약"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${BOLD} 논문 Table 1 (CLIPSIM + Consistency + FVD)       ${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
python -c "
import csv, sys
path = '$RESULTS/table1_full.csv'
try:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'  {\"Model\":<30}  {\"CLIPSIM\":>8}  {\"Consist\":>8}  {\"Drop\":>8}  {\"FVD\":>8}')
    print('  ' + '-' * 65)
    for r in rows:
        print(f'  {r[\"model\"]:<30}  {float(r[\"clipsim\"]):>8.4f}  {float(r[\"consistency\"]):>8.4f}  {float(r[\"consistency_drop\"]):>8.4f}  {float(r[\"fvd\"]):>8.2f}')
except Exception as e:
    print(f'  결과 파일 읽기 실패: {e}')
"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${BOLD} 논문 Table 2 (Scaling Law: K vs VRAM/Time)       ${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
python -c "
import csv
path = '$RESULTS/table2_scaling.csv'
try:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'  {\"K\":>4}  {\"QSFM(ms)\":>10}  {\"QSFM VRAM\":>10}  {\"Attn(ms)\":>10}  {\"Attn VRAM\":>10}')
    print('  ' + '-' * 55)
    for r in rows:
        print(f'  {r[\"K\"]:>4}  {float(r[\"qsfm_time_ms\"]):>10.1f}  {float(r[\"qsfm_vram_mb\"]):>9.1f}M  {float(r[\"attn_time_ms\"]):>10.3f}  {float(r[\"attn_vram_mb\"]):>9.1f}M')
except Exception as e:
    print(f'  결과 파일 읽기 실패: {e}')
"

echo ""
echo -e "${GREEN}${BOLD}모든 실험 완료!${NC}"
echo "  완료 시각: $(date)"
echo "  결과 위치: $RESULTS/"
echo ""
echo "  파일 목록:"
ls -lh $RESULTS/ 2>/dev/null || echo "  (결과 없음)"
