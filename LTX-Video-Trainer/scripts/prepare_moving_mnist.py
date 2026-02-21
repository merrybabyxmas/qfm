#!/usr/bin/env python
"""
Moving MNIST 데이터셋 준비  (QSFM Tier 1: Scalability Diagnostic)
=================================================================
목적: 픽셀 디테일 없이 "K가 32/64개로 늘어도 QSFM은 O(1)로 생성"을 보여주는
      투명한 확장성 실험용 데이터셋.

각 시퀀스는 20 프레임의 2개 이동하는 필기 숫자.
→ 단순한 도형 움직임이라 QSFM의 시간적 일관성 이점을 명확히 측정 가능.

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    python scripts/prepare_moving_mnist.py \\
        --output_dir /home/dongwoo43/qfm/mnist_data \\
        --n_samples 64

전처리 (Moving MNIST용 해상도 버킷):
    PYTHONPATH=src python scripts/preprocess_dataset.py \\
        --dataset_path /home/dongwoo43/qfm/mnist_data/dataset.json \\
        --output_dir /home/dongwoo43/qfm/mnist_data \\
        --resolution_buckets "64x64x20" \\
        --id_token "MNIST" \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"

# 다양성을 위한 캡션 풀 (n_samples에 순환 적용)
CAPTION_POOL = [
    "two animated digits moving across a black background",
    "handwritten numbers bouncing around a dark screen in random walk",
    "two white digits sliding diagonally on a black canvas",
    "moving handwritten digits in a simple looping animation",
    "two numbers performing a random walk on a dark screen",
    "white MNIST digits in motion on black background",
    "pair of handwritten digits bouncing off screen edges",
    "animated handwritten number sequence on dark background",
    "two moving digit objects in a synthetic video clip",
    "simple digit motion sequence for scalability testing",
    "digits number zero and three gliding across screen",
    "handwritten one and seven digits in continuous motion",
    "digits bouncing with momentum in an animated scene",
    "two-digit sequence in random trajectory on black field",
    "synthetic moving MNIST sequence for video generation",
    "minimalist digit animation on black background canvas",
]


def download_moving_mnist(dest_dir: Path) -> Path:
    """Moving MNIST npy 파일 다운로드 (10000시퀀스 × 20프레임 × 64×64)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "mnist_test_seq.npy"
    if out.exists():
        print(f"  이미 다운로드됨: {out}")
        return out
    print(f"  다운로드 중: {MNIST_URL}")
    print("  (파일 크기 약 780MB, 시간이 걸릴 수 있습니다)")
    urllib.request.urlretrieve(MNIST_URL, str(out))
    print(f"  저장 완료: {out}")
    return out


def seq_to_mp4(seq: np.ndarray, out_path: Path, fps: int = 10) -> None:
    """
    (T, H, W) uint8 그레이스케일 시퀀스 → MP4.

    Args:
        seq      : (T, H, W) uint8  0~255
        out_path : 저장 경로
        fps      : 초당 프레임 수
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 미설치. 실행: pip install opencv-python")

    T, H, W = seq.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H), isColor=True)
    for t in range(T):
        frame_gray = seq[t]
        frame_rgb = np.stack([frame_gray, frame_gray, frame_gray], axis=-1)
        writer.write(frame_rgb)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moving MNIST → QSFM 학습용 dataset.json + MP4 준비"
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("/home/dongwoo43/qfm/mnist_data"),
        help="데이터 저장 루트 디렉토리"
    )
    parser.add_argument(
        "--n_samples", type=int, default=64,
        help="사용할 시퀀스 수 (기본 64; K=32 학습시 최소 32 필요)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="MP4 저장 FPS (기본 10)"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="npy 파일 이미 있을 때 다운로드 스킵"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Moving MNIST Dataset Preparation  (QSFM Tier 1)")
    print(f"  output_dir : {args.output_dir}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  fps        : {args.fps}")
    print("=" * 65)

    # ── 다운로드 ─────────────────────────────────────────────────────
    raw_dir = args.output_dir / "raw"
    if args.skip_download and (raw_dir / "mnist_test_seq.npy").exists():
        npy_path = raw_dir / "mnist_test_seq.npy"
        print(f"다운로드 스킵: {npy_path}")
    else:
        npy_path = download_moving_mnist(raw_dir)

    # ── 로드 ─────────────────────────────────────────────────────────
    print("npy 로딩 중...")
    data = np.load(str(npy_path))          # (T=20, N=10000, H=64, W=64) uint8
    T_total, N_total, H, W = data.shape
    print(f"  Shape: (T={T_total}, N={N_total}, H={H}, W={W})")

    n = min(args.n_samples, N_total)
    scenes_dir = args.output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # ── 시퀀스 → MP4 변환 ─────────────────────────────────────────────
    print(f"\n{n}개 시퀀스 MP4 변환 중...")
    dataset: list[dict] = []

    for i in range(n):
        seq = data[:, i, :, :]   # (T, H, W) uint8

        vid_name = f"moving_mnist_{i:04d}.mp4"
        vid_path = scenes_dir / vid_name
        seq_to_mp4(seq, vid_path, fps=args.fps)

        caption = CAPTION_POOL[i % len(CAPTION_POOL)]
        dataset.append({"caption": caption, "media_path": str(vid_path)})

        if (i + 1) % 16 == 0 or i == n - 1:
            print(f"  {i+1:4d} / {n:4d}  [{vid_path.name}]")

    # ── dataset.json 저장 ─────────────────────────────────────────────
    json_path = args.output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 완료!")
    print(f"  {n}개 시퀀스 저장 위치 : {scenes_dir}")
    print(f"  dataset.json          : {json_path}")
    print()
    print("─" * 65)
    print("다음 단계: 전처리 (LTX-Video VAE 인코딩)")
    print("─" * 65)
    print(f"""
conda activate afm
cd /home/dongwoo43/qfm/LTX-Video-Trainer
PYTHONPATH=src python scripts/preprocess_dataset.py \\
    --dataset_path {json_path} \\
    --output_dir {args.output_dir} \\
    --resolution_buckets "64x64x20" \\
    --id_token "MNIST" \\
    --device cuda
""")
    print("─" * 65)
    print("전처리 후 QSFM 학습 (mnist_qsfm.yaml 참고):")
    print("─" * 65)
    print("""
conda activate afm
cd /home/dongwoo43/qfm/LTX-Video-Trainer
python scripts/train.py --config configs/qsfm_mnist_lora.yaml
""")


if __name__ == "__main__":
    main()
