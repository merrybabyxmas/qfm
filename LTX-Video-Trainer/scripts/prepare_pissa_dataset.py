#!/usr/bin/env python
"""
fxmeng/pissa-dataset → QSFM 학습 데이터셋 변환
================================================
HuggingFace의 fxmeng/pissa-dataset을 다운로드하여
QSFM 학습 형식(dataset.json + 합성 비디오)으로 변환.

fxmeng/pissa-dataset:
  - instruction/output 쌍 (텍스트 instruction-following 데이터)
  - 844K 샘플 / 26가지 타입

여기서는:
  1. instruction 텍스트를 비디오 캡션으로 사용
  2. 각 캡션에 대응하는 합성 비디오 생성 (cv2 기반)
     - 텍스트와 패턴을 비디오에 시각화
  3. dataset.json 형식으로 저장 → preprocess_dataset.py 입력

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    python scripts/prepare_pissa_dataset.py \\
        --output_dir /home/dongwoo43/qfm/pissa_data \\
        --n_samples 40 \\
        --type_filter math code  # 특정 타입 필터 (선택)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np


# ── 기본 설정 ────────────────────────────────────────────────────────────────
DEFAULT_TYPES = [
    "math", "code", "dyck_languages", "logical_deduction", "common_sense"
]

# instruction을 짧은 캡션으로 변환하는 템플릿
CAPTION_TEMPLATES = {
    "math":              "a mathematical problem-solving sequence with numbers",
    "code":              "a computer programming code execution animation",
    "dyck_languages":    "bracket and symbol sequence completion visualization",
    "logical_deduction": "logical reasoning and deduction process animation",
    "common_sense":      "common sense reasoning scenario visualization",
    "default":           "an instruction-following task visualization",
}


def _make_caption(instruction: str, type_: str) -> str:
    """instruction → 짧은 시각적 캡션."""
    # 타입 기반 캡션 + instruction 첫 10단어
    base = CAPTION_TEMPLATES.get(type_, CAPTION_TEMPLATES["default"])
    words = instruction.split()[:10]
    snippet = " ".join(words).rstrip(".,;:")
    return f"{base}: {snippet}"


def generate_synthetic_video(
    caption: str,
    type_: str,
    out_path: Path,
    width: int = 320,
    height: int = 240,
    n_frames: int = 48,
    fps: int = 8,
) -> bool:
    """
    PISSA instruction에 대응하는 합성 비디오 생성.
    텍스트, 움직이는 그래프/패턴으로 instruction을 시각화.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), True)

    # 타입별 배경 색상 시드
    color_seed = abs(hash(type_)) % 0xFFFFFF
    r = (color_seed >> 16) & 0xFF
    g = (color_seed >> 8) & 0xFF
    b = color_seed & 0xFF

    # caption을 여러 줄로 분리
    wrapped = textwrap.wrap(caption[:80], width=38)

    for t in range(n_frames):
        progress = t / max(n_frames - 1, 1)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 배경: 어두운 그라디언트
        for y in range(height):
            frame[y, :] = (
                int(15 + 25 * y / height),
                int(15 + 15 * y / height),
                int(20 + 30 * y / height),
            )

        # ── 패턴 1: 움직이는 파형 (수학/논리 시각화) ──
        for x in range(0, width, 2):
            amplitude = height * 0.12
            freq = 3.0 + 2.0 * (abs(hash(type_)) % 5)
            phase = progress * 2 * np.pi * freq
            y_val = int(height * 0.35 + amplitude * np.sin(x * 0.05 + phase))
            y_val = max(0, min(height - 1, y_val))
            cv2.circle(frame, (x, y_val), 1, (b, g, r), -1)

        # ── 패턴 2: 진행 바 (completion progress) ──
        bar_w = int(width * 0.85 * progress)
        bar_x = int(width * 0.075)
        bar_y = int(height * 0.72)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(width * 0.85), bar_y + 8),
                      (40, 40, 40), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (r // 2, g, b), -1)

        # ── 패턴 3: 움직이는 노드 그래프 (논리 연결) ──
        n_nodes = min(4, len(type_))
        for ni in range(n_nodes):
            nx_ = int(width * (0.15 + 0.7 * ni / max(n_nodes - 1, 1)))
            ny_ = int(height * 0.55 + 15 * np.sin(progress * np.pi + ni * 1.5))
            cv2.circle(frame, (nx_, ny_), 6, (r, g // 2, b // 2), -1)
            if ni < n_nodes - 1:
                nx2 = int(width * (0.15 + 0.7 * (ni + 1) / max(n_nodes - 1, 1)))
                ny2 = int(height * 0.55 + 15 * np.sin(progress * np.pi + (ni + 1) * 1.5))
                alpha = int(100 + 100 * np.sin(progress * np.pi * 2 + ni))
                cv2.line(frame, (nx_, ny_), (nx2, ny2), (alpha, alpha // 2, 200 - alpha), 1)

        # ── 텍스트 오버레이 ──
        cv2.putText(frame, type_[:20], (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        for li, line in enumerate(wrapped[:3]):
            cv2.putText(frame, line, (8, 88 + li * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        # 프레임 번호 (오른쪽 하단)
        cv2.putText(frame, f"{t+1:02d}/{n_frames}",
                    (width - 50, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        writer.write(frame)

    writer.release()
    return out_path.exists()


def download_and_prepare(
    output_dir: Path,
    n_samples: int = 40,
    type_filter: list[str] | None = None,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """
    fxmeng/pissa-dataset 다운로드 → QSFM dataset.json 생성.

    Returns:
        dataset 항목 리스트 [{'caption': ..., 'media_path': ...}, ...]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets")

    print(f"  HuggingFace 데이터셋 로딩: fxmeng/pissa-dataset")
    print(f"  split={split}, n_samples={n_samples}")

    ds = load_dataset("fxmeng/pissa-dataset", split=split, streaming=True)

    # 타입 필터 적용
    if type_filter:
        ds = ds.filter(lambda x: x.get("type", "") in type_filter)

    # n_samples 수집 (셔플)
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    samples = []
    for item in ds:
        if len(samples) >= n_samples:
            break
        inst = item.get("instruction", "")
        type_ = item.get("type", "default")
        if inst.strip():
            samples.append({"instruction": inst, "type": type_})

    if not samples:
        raise RuntimeError(f"샘플 수집 실패. type_filter={type_filter}")

    print(f"  {len(samples)}개 샘플 수집 완료")
    print(f"  타입 분포: {dict((s['type'], sum(1 for x in samples if x['type']==s['type'])) for s in samples[:1])}")

    # 합성 비디오 생성
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    for i, sample in enumerate(samples):
        type_ = sample["type"]
        inst = sample["instruction"]
        caption = _make_caption(inst, type_)

        vid_name = f"pissa_{type_[:12]}_{i:04d}.mp4"
        vid_path = scenes_dir / vid_name

        if not vid_path.exists():
            ok = generate_synthetic_video(caption, type_, vid_path)
            if not ok:
                print(f"  [warn] {vid_name} 생성 실패 (스킵)")
                continue

        dataset.append({
            "caption": caption,
            "media_path": str(vid_path),
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)} 완료")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="fxmeng/pissa-dataset → QSFM 학습 데이터셋 변환"
    )
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/pissa_data"))
    parser.add_argument("--n_samples", type=int, default=40,
                        help="사용할 샘플 수 (기본 40)")
    parser.add_argument("--type_filter", nargs="*", default=None,
                        help="특정 타입만 사용 (예: math code). 기본=전체")
    parser.add_argument("--split",  default="train")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--width",  type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PISSA Dataset → QSFM 형식 변환")
    print(f"  HuggingFace : fxmeng/pissa-dataset")
    print(f"  출력 경로   : {args.output_dir}")
    print(f"  샘플 수     : {args.n_samples}")
    print(f"  타입 필터   : {args.type_filter or '전체'}")
    print("=" * 65)

    dataset = download_and_prepare(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        type_filter=args.type_filter,
        split=args.split,
        seed=args.seed,
    )

    json_path = args.output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 완료!")
    print(f"  {len(dataset)}개 클립 생성")
    print(f"  dataset.json: {json_path}")
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
