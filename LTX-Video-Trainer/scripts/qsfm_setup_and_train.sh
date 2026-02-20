#!/usr/bin/env bash
# ============================================================
# QSFM 학습 전체 파이프라인 셋업 스크립트
# 1. 긴 공개 도메인 비디오 다운로드
# 2. 씬 분할 (4개 샷)
# 3. 자동 캡션 생성
# 4. 잠재 벡터 전처리
# 5. QSFM 학습 실행
# ============================================================

set -e
CONDA_ENV=afm
DATA_DIR=/home/dongwoo43/qfm/qsfm_data
TRAINER_DIR=/home/dongwoo43/qfm/LTX-Video-Trainer

echo "============================================================"
echo " QSFM 학습 파이프라인 시작"
echo "============================================================"

# ── Step 0 : 디렉토리 준비 ────────────────────────────────
mkdir -p "$DATA_DIR/videos" "$DATA_DIR/scenes"

# ── Step 1 : 긴 공개 비디오 다운로드 ─────────────────────
VIDEO_URL="https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_surround-fix.avi"
RAW_VIDEO="$DATA_DIR/videos/big_buck_bunny.mp4"

if [ ! -f "$RAW_VIDEO" ]; then
    echo "[1/5] Downloading Big Buck Bunny (공개 도메인 영상)..."
    conda run -n $CONDA_ENV python -c "
import urllib.request, sys
url = '$VIDEO_URL'
out = '$RAW_VIDEO'
print(f'  URL: {url}')
print(f'  Saving to: {out}')
# Use yt-dlp if available, otherwise urllib
import subprocess
result = subprocess.run(
    ['yt-dlp', '-o', out, '--no-playlist', '-x', '--audio-format', 'mp4', url],
    capture_output=True
)
if result.returncode != 0:
    # fallback to direct download via requests
    import requests
    resp = requests.get(url, stream=True, timeout=60)
    with open(out, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
print('  Download complete!')
" 2>&1 || {
        echo "[1/5] yt-dlp 실패 → wget 으로 재시도..."
        wget -q --show-progress -O "$RAW_VIDEO" "$VIDEO_URL" || {
            echo "[1/5] wget 도 실패 → 샘플 비디오 직접 다운로드..."
            # 더 작은 샘플 비디오
            conda run -n $CONDA_ENV python -c "
import urllib.request
urls = [
    'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4',
]
import os
for url in urls:
    try:
        print(f'Trying: {url}')
        urllib.request.urlretrieve(url, '$RAW_VIDEO')
        size = os.path.getsize('$RAW_VIDEO')
        if size > 100000:
            print(f'Downloaded: {size/1e6:.1f} MB')
            break
    except Exception as e:
        print(f'Failed: {e}')
        continue
"
        }
    }
fi

echo "[1/5] 비디오 준비 완료: $RAW_VIDEO"

# ── Step 2 : 씬 분할 ─────────────────────────────────────
SCENES_DIR="$DATA_DIR/scenes"
echo "[2/5] 씬 분할 중 (최소 5초 이상 클립 4개 이상)..."
conda run -n $CONDA_ENV python "$TRAINER_DIR/scripts/split_scenes.py" \
    "$RAW_VIDEO" \
    "$SCENES_DIR" \
    --filter-shorter-than 5s \
    --max-scenes 8 2>&1 || {
    echo "[2/5] 자동 씬 분할 실패 → ffmpeg 으로 수동 분할..."
    conda run -n $CONDA_ENV python -c "
import subprocess, os

video = '$RAW_VIDEO'
out_dir = '$SCENES_DIR'
os.makedirs(out_dir, exist_ok=True)

# 비디오 길이 확인
result = subprocess.run(
    ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
     '-of', 'default=noprint_wrappers=1:nokey=1', video],
    capture_output=True, text=True
)
duration = float(result.stdout.strip() or '600')
print(f'Video duration: {duration:.1f}s')

# 4개 씬으로 균등 분할 (각 최소 15초)
n_scenes = min(4, max(1, int(duration / 15)))
seg_len = duration / n_scenes

for i in range(n_scenes):
    start = i * seg_len
    out_file = os.path.join(out_dir, f'scene_{i:04d}.mp4')
    cmd = [
        'ffmpeg', '-y', '-ss', str(start), '-i', video,
        '-t', str(seg_len),
        '-vf', 'scale=512:320:force_original_aspect_ratio=decrease,pad=512:320:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-an', out_file
    ]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(out_file):
        size = os.path.getsize(out_file)
        print(f'  Created: {out_file} ({size/1e6:.1f} MB)')
"
}

# 씬 수 확인
N_SCENES=$(ls "$SCENES_DIR"/*.mp4 2>/dev/null | wc -l)
echo "[2/5] 생성된 씬: ${N_SCENES}개"

if [ "$N_SCENES" -lt 4 ]; then
    echo "[오류] 최소 4개 씬 필요 (현재: $N_SCENES)"
    exit 1
fi

# ── Step 3 : 캡션 생성 ──────────────────────────────────
echo "[3/5] 캡션 생성 중..."
conda run -n $CONDA_ENV python "$TRAINER_DIR/scripts/caption_videos.py" \
    "$SCENES_DIR" \
    --output "$DATA_DIR/dataset.json" \
    --use-8bit 2>&1 || {
    echo "[3/5] 자동 캡션 실패 → 기본 캡션으로 대체..."
    conda run -n $CONDA_ENV python -c "
import json, os, glob

scenes = sorted(glob.glob('$SCENES_DIR/*.mp4'))[:8]
data = [
    {'caption': f'A video scene {i+1} with natural motion and visual content',
     'media_path': p}
    for i, p in enumerate(scenes)
]
with open('$DATA_DIR/dataset.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Created dataset.json with {len(data)} entries')
for d in data:
    print(f'  - {os.path.basename(d[\"media_path\"])}: {d[\"caption\"]}')
"
}

echo "[3/5] dataset.json 생성 완료"

# ── Step 4 : 잠재 벡터 전처리 ───────────────────────────
echo "[4/5] 잠재 벡터 및 텍스트 임베딩 전처리 중..."
mkdir -p "$DATA_DIR/.precomputed"

conda run -n $CONDA_ENV python "$TRAINER_DIR/scripts/preprocess_dataset.py" \
    "$DATA_DIR/dataset.json" \
    --resolution-buckets "512x320x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --model-source "LTXV_2B_0.9.6_DEV" \
    --device "cuda" \
    --load-text-encoder-in-8bit \
    --output-dir "$DATA_DIR" 2>&1

echo "[4/5] 전처리 완료"

# ── Step 5 : QSFM 학습 ──────────────────────────────────
echo "[5/5] QSFM 학습 시작..."
conda run -n $CONDA_ENV python "$TRAINER_DIR/scripts/train.py" \
    "$TRAINER_DIR/configs/qsfm_2b_lora.yaml" 2>&1

echo "============================================================"
echo " QSFM 학습 파이프라인 완료!"
echo " 결과물: $TRAINER_DIR/outputs/qsfm_lora/"
echo "============================================================"
