#!/usr/bin/env bash
# =============================================================================
# QSFM Full Experiment Pipeline  (논문 전체 실험 자동화)
# =============================================================================
# 이 스크립트는 QSFM 논문 실험 전체를 순차적으로 실행합니다:
#
#  PHASE 0: 환경 점검
#  PHASE 1: 데이터셋 준비
#           1a. Big Buck Bunny (이미 완료)
#           1b. Moving MNIST  (Tier 1)
#           1c. UCF-101       (Tier 2, 합성 모드)
#  PHASE 2: LTX-Video VAE 전처리
#           2a. Moving MNIST 전처리
#           2b. UCF-101 전처리
#  PHASE 3: QSFM 학습 (데이터셋별)
#           3a. Big Buck Bunny (기존 데이터, K=4)
#           3b. Moving MNIST  (K=4)
#           3c. UCF-101       (K=4)
#  PHASE 4: LTX-Video Pure Baseline 추론
#  PHASE 5: Table 2: Scaling Law Benchmark
#  PHASE 6: 평가 워크스페이스 업데이트
#  PHASE 7: Table 1: 통합 평가 (CLIPSIM + Temporal Consistency + FVD)
#  PHASE 8: 결과 요약 출력
#
# 사용법:
#   conda activate afm
#   cd /home/dongwoo43/qfm/LTX-Video-Trainer
#   bash scripts/run_full_experiment_pipeline.sh [OPTIONS]
#
# 옵션:
#   --skip-training      학습 단계 스킵 (이미 완료된 경우)
#   --skip-baseline      LTX-Video Pure 추론 스킵
#   --skip-mnist         Moving MNIST 단계 스킵
#   --skip-ucf           UCF-101 단계 스킵
#   --quick              빠른 테스트 (steps=100, K=4)
#   --steps N            학습 스텝 수 (기본 1000)
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

# ── 옵션 파싱 ────────────────────────────────────────────────────────────────
SKIP_TRAINING=false
SKIP_BASELINE=false
SKIP_MNIST=false
SKIP_UCF=false
TRAIN_STEPS=1000
QUICK=false

for arg in "$@"; do
    case $arg in
        --skip-training) SKIP_TRAINING=true ;;
        --skip-baseline) SKIP_BASELINE=true ;;
        --skip-mnist)    SKIP_MNIST=true ;;
        --skip-ucf)      SKIP_UCF=true ;;
        --quick)         QUICK=true; TRAIN_STEPS=200 ;;
        --steps)         shift; TRAIN_STEPS=$1 ;;
    esac
done

if $QUICK; then
    warn "QUICK 모드: steps=${TRAIN_STEPS}, 빠른 파이프라인 테스트"
fi

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE=/home/dongwoo43/qfm
TRAINER=$BASE/LTX-Video-Trainer
DATA_BBB=$BASE/qsfm_data
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
phase "PHASE 1a: Big Buck Bunny 데이터 확인"
# ─────────────────────────────────────────────────────────────────────────────

if [ -f "$DATA_BBB/dataset.json" ]; then
    N_BBB=$(python -c "import json; print(len(json.load(open('$DATA_BBB/dataset.json'))))")
    ok "Big Buck Bunny: $N_BBB 씬 준비됨"
else
    warn "Big Buck Bunny 데이터 없음 (qsfm_setup_and_train.sh 먼저 실행 필요)"
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
if ! $SKIP_MNIST && [ -f "$DATA_MNIST/dataset.json" ]; then
phase "PHASE 2a: Moving MNIST VAE 전처리"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$DATA_MNIST/latents" ]; then
        ok "Moving MNIST latents 이미 존재 (스킵)"
    else
        log "Moving MNIST VAE 인코딩..."
        python scripts/preprocess_dataset.py \
            --dataset_path $DATA_MNIST/dataset.json \
            --output_dir $DATA_MNIST \
            --resolution_buckets "64x64x20" \
            --device cuda
        ok "Moving MNIST 전처리 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_UCF && [ -f "$DATA_UCF/dataset.json" ]; then
phase "PHASE 2b: UCF-101 VAE 전처리"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$DATA_UCF/latents" ]; then
        ok "UCF-101 latents 이미 존재 (스킵)"
    else
        log "UCF-101 VAE 인코딩..."
        python scripts/preprocess_dataset.py \
            --dataset_path $DATA_UCF/dataset.json \
            --output_dir $DATA_UCF \
            --resolution_buckets "320x240x48" \
            --device cuda
        ok "UCF-101 전처리 완료"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
if ! $SKIP_TRAINING; then
phase "PHASE 3a: QSFM 학습 — Big Buck Bunny (K=4)"
# ─────────────────────────────────────────────────────────────────────────────
    if [ -d "$TRAINER/outputs/qsfm_lora/checkpoints" ]; then
        ok "Big Buck Bunny QSFM 체크포인트 이미 존재"
    else
        log "QSFM 학습 시작 (steps=$TRAIN_STEPS)..."
        python scripts/train.py --config configs/qsfm_2b_lora.yaml \
            optimization.steps=$TRAIN_STEPS
        ok "Big Buck Bunny QSFM 학습 완료"
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
