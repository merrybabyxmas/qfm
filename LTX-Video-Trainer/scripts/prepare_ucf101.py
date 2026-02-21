#!/usr/bin/env python
"""
UCF-101 데이터셋 준비  (QSFM Tier 2: Standard Benchmark)
==========================================================
UCF-101에서 N개 클립을 자동 다운로드하여 QSFM 전처리 형식으로 변환.

UCF-101 공식 출처: https://www.crcv.ucf.edu/data/UCF101.php
용량: 전체 6.5GB (RAR)

이 스크립트는 두 가지 모드를 지원합니다:
  1. --subset : 클래스별 소수 클립만 다운로드 (빠름, 비교 실험용)
  2. --full   : 전체 6.5GB 다운로드 (FVD 벤치마크 정밀도 높음)

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer

    # 빠른 테스트 (5개 클래스 × 8 클립 = 40개)
    python scripts/prepare_ucf101.py \\
        --output_dir /home/dongwoo43/qfm/ucf101_data \\
        --n_classes 5 \\
        --clips_per_class 8

    # 전처리 (이후)
    PYTHONPATH=src python scripts/preprocess_dataset.py \\
        --dataset_path /home/dongwoo43/qfm/ucf101_data/dataset.json \\
        --output_dir /home/dongwoo43/qfm/ucf101_data \\
        --resolution_buckets "320x240x48" \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# UCF-101 공식 클래스 목록 중 대표 5개 (동물·자연 관련 → QSFM 데이터셋과 유사 도메인)
UCF_SUBSET_CLASSES = [
    "HorseRiding",
    "WalkingWithDog",
    "JavelinThrow",
    "HighJump",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Bowling",
    "BoxingPunchingBag",
    "CliffDiving",
    "Diving",
    "Fencing",
    "GolfSwing",
    "Hammering",
    "HandstandWalking",
    "IceDancing",
    "JugglingBalls",
    "LongJump",
    "MilitaryParade",
    "Rowing",
]

# 각 클래스 캡션 (자동 생성)
CLASS_CAPTIONS = {
    "HorseRiding":       "a person riding a horse outdoors",
    "WalkingWithDog":    "a person walking with a dog on a leash",
    "JavelinThrow":      "an athlete throwing a javelin in a field",
    "HighJump":          "an athlete performing a high jump",
    "BasketballDunk":    "a basketball player dunking the ball",
    "BenchPress":        "a person doing bench press with weights",
    "Biking":            "a person riding a bicycle on a road",
    "Bowling":           "a person bowling in a bowling alley",
    "BoxingPunchingBag": "a boxer punching a heavy bag",
    "CliffDiving":       "a person cliff diving into water",
    "Diving":            "a diver performing a dive into a pool",
    "Fencing":           "two fencers competing in a match",
    "GolfSwing":         "a golfer swinging a golf club",
    "Hammering":         "a person hammering a nail into wood",
    "HandstandWalking":  "a person walking on their hands",
    "IceDancing":        "ice dancers performing a routine",
    "JugglingBalls":     "a juggler juggling multiple balls",
    "LongJump":          "an athlete performing a long jump",
    "MilitaryParade":    "soldiers marching in a military parade",
    "Rowing":            "athletes rowing a boat on water",
}

# UCF-101 공식 클래스별 비디오 URL 패턴
UCF101_BASE_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
# 개별 클래스 AVI 파일이 들어있는 RAR 압축 파일 (ZIP 기반)
# 공식 분할 파일 리스트 (텍스트)
SPLIT_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


def try_download_subset_via_rar(
    output_dir: Path,
    n_classes: int,
    clips_per_class: int,
) -> bool:
    """
    rar 유틸리티를 사용한 선택적 클래스 추출.
    rar 가 없으면 False 반환.
    """
    if not shutil.which("rar") and not shutil.which("unrar"):
        return False

    rar_path = output_dir / "UCF101.rar"
    if not rar_path.exists():
        print(f"  UCF-101 다운로드 중 ({UCF101_BASE_URL})")
        print("  ⚠ 전체 6.5GB 파일입니다. 시간이 걸릴 수 있습니다.")
        urllib.request.urlretrieve(UCF101_BASE_URL, str(rar_path))
        print(f"  완료: {rar_path}")

    selected = UCF_SUBSET_CLASSES[:n_classes]
    extract_dir = output_dir / "UCF-101"
    extract_dir.mkdir(parents=True, exist_ok=True)

    unrar = shutil.which("unrar") or shutil.which("rar")
    for cls in selected:
        cmd = [unrar, "e", str(rar_path), f"UCF-101/{cls}/*", str(extract_dir / cls), "-y"]
        subprocess.run(cmd, capture_output=True)

    return True


def generate_synthetic_ucf_like(
    output_dir: Path,
    n_classes: int,
    clips_per_class: int,
    frames: int = 48,
    h: int = 240,
    w: int = 320,
) -> list[dict]:
    """
    UCF-101 없을 때 동작 유사한 합성 비디오 생성 (파이프라인 테스트용).
    실제 논문 실험에는 진짜 UCF-101 사용 필요.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("pip install opencv-python numpy")

    print(f"\n  ⚠ UCF-101 다운로드 스킵 → 합성 동작 비디오 생성 (파이프라인 테스트용)")
    print(f"  실제 논문 실험 시 --full 옵션 또는 수동 다운로드 필요")

    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    classes = UCF_SUBSET_CLASSES[:n_classes]

    for cls in classes:
        caption = CLASS_CAPTIONS.get(cls, f"person performing {cls}")
        for i in range(clips_per_class):
            # 합성 비디오: 움직이는 도형으로 동작 시뮬레이션
            vid_name = f"ucf_{cls}_{i:03d}.mp4"
            vid_path = scenes_dir / vid_name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 25, (w, h), True)

            # 클래스별 고유 색상 (재현성)
            color_seed = hash(cls) % 0xFFFFFF
            r = (color_seed >> 16) & 0xFF
            g = (color_seed >> 8) & 0xFF
            b = color_seed & 0xFF

            for t in range(frames):
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                bg[:] = (20, 20, 40)  # 어두운 배경

                # 움직이는 원 (동작 시뮬레이션)
                cx = int(w * 0.2 + (w * 0.6) * (t / frames))
                cy = int(h * 0.5 + h * 0.2 * np.sin(t * 0.3))
                cv2.circle(bg, (cx, cy), 30, (b, g, r), -1)

                # 두 번째 원 (상호작용)
                cx2 = int(w * 0.8 - (w * 0.4) * (t / frames))
                cy2 = int(h * 0.4 + h * 0.15 * np.cos(t * 0.4))
                cv2.circle(bg, (cx2, cy2), 20, (r // 2, g // 2, b), -1)

                # 클래스 텍스트
                cv2.putText(bg, cls[:12], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                writer.write(bg)
            writer.release()

            dataset.append({"caption": caption, "media_path": str(vid_path)})

        print(f"    {cls}: {clips_per_class}개 생성")

    return dataset


def download_real_ucf101(
    output_dir: Path,
    n_classes: int,
    clips_per_class: int,
) -> list[dict] | None:
    """
    실제 UCF-101 다운로드 시도.
    성공 시 dataset.json 항목 반환, 실패 시 None 반환.
    """
    # unrar 체크
    if not (shutil.which("unrar") or shutil.which("rar")):
        print("  unrar 없음. 설치: sudo apt-get install unrar")
        return None

    rar_path = output_dir / "UCF101.rar"
    if not rar_path.exists():
        print(f"  UCF-101 다운로드 시작...")
        print(f"  크기: ~6.5GB, 시간이 필요합니다.")
        try:
            urllib.request.urlretrieve(UCF101_BASE_URL, str(rar_path))
        except Exception as e:
            print(f"  다운로드 실패: {e}")
            return None

    # 선택 클래스 추출
    selected = UCF_SUBSET_CLASSES[:n_classes]
    extract_dir = output_dir / "UCF-101"
    extract_dir.mkdir(parents=True, exist_ok=True)

    unrar_cmd = shutil.which("unrar") or shutil.which("rar")
    dataset = []
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    for cls in selected:
        cls_dir = extract_dir / cls
        subprocess.run(
            [unrar_cmd, "e", str(rar_path), f"UCF-101/{cls}/*.avi",
             str(cls_dir), "-y"],
            capture_output=True,
        )
        avi_files = list(cls_dir.glob("*.avi"))[:clips_per_class]
        caption = CLASS_CAPTIONS.get(cls, f"person performing {cls}")
        for avi in avi_files:
            mp4 = scenes_dir / (avi.stem + ".mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", str(mp4)],
                capture_output=True,
            )
            if mp4.exists():
                dataset.append({"caption": caption, "media_path": str(mp4)})

    return dataset if dataset else None


def main():
    parser = argparse.ArgumentParser(description="UCF-101 데이터셋 준비")
    parser.add_argument("--output_dir",      type=Path,
                        default=Path("/home/dongwoo43/qfm/ucf101_data"))
    parser.add_argument("--n_classes",        type=int, default=5,
                        help="사용할 클래스 수 (기본 5)")
    parser.add_argument("--clips_per_class",  type=int, default=8,
                        help="클래스당 클립 수 (기본 8)")
    parser.add_argument("--full",             action="store_true",
                        help="전체 UCF-101 다운로드 시도")
    parser.add_argument("--synthetic_only",   action="store_true",
                        help="합성 비디오만 생성 (빠른 파이프라인 테스트)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("UCF-101 Dataset Preparation  (QSFM Tier 2)")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  n_classes        : {args.n_classes}")
    print(f"  clips_per_class  : {args.clips_per_class}")
    print(f"  mode             : {'full' if args.full else 'synthetic' if args.synthetic_only else 'auto'}")
    print("=" * 65)

    # ── 데이터 취득 ─────────────────────────────────────────────────
    dataset = None

    if not args.synthetic_only and args.full:
        print("\n실제 UCF-101 다운로드 시도...")
        dataset = download_real_ucf101(args.output_dir, args.n_classes, args.clips_per_class)

    if dataset is None:
        dataset = generate_synthetic_ucf_like(
            args.output_dir, args.n_classes, args.clips_per_class
        )

    # ── dataset.json 저장 ────────────────────────────────────────────
    json_path = args.output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total = len(dataset)
    print(f"\n✅ 완료!")
    print(f"  {total}개 클립 준비 ({args.n_classes}클래스 × {args.clips_per_class}클립)")
    print(f"  dataset.json : {json_path}")
    print()
    print("─" * 65)
    print("다음: LTX-Video VAE 전처리")
    print("─" * 65)
    print(f"""
conda activate afm
cd /home/dongwoo43/qfm/LTX-Video-Trainer
PYTHONPATH=src python scripts/preprocess_dataset.py \\
    --dataset_path {json_path} \\
    --output_dir {args.output_dir} \\
    --resolution_buckets "320x240x48" \\
    --device cuda
""")


if __name__ == "__main__":
    main()
