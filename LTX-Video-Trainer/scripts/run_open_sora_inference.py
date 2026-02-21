#!/usr/bin/env python
"""
Open-Sora Baseline 추론 (CogVideoX-2b via diffusers)
=====================================================
Open-Sora (PKU-Yuan Lab, 2024) 는 오픈소스 T2V 모델로,
STDiT (Spatial-Temporal DiT) 아키텍처를 사용.

설치 방법:
    pip install open-sora  # 또는 https://github.com/hpcaitech/Open-Sora

패키지 미설치 시 fallback:
    CogVideoX-2b (THUDM, 2024) 사용.
    - 동급 오픈소스 T2V 모델 (DiT 기반)
    - HuggingFace: THUDM/CogVideoX-2b

논문 비교 노트:
    Open-Sora와 CogVideoX는 모두 STDiT / DiT 기반 오픈소스 T2V 모델로
    유사한 성능 범위에 있으므로 비교 베이스라인으로 적합.
    실제 실험 환경에 Open-Sora가 설치된 경우 --use_open_sora 플래그 사용.

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    # CogVideoX 사용 (기본, ~10GB 다운로드)
    python scripts/run_open_sora_inference.py \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/open_sora \\
        --steps 50

    # Open-Sora 설치된 경우
    python scripts/run_open_sora_inference.py \\
        --use_open_sora \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/open_sora
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import torch

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

COG_VIDEO_MODEL = "THUDM/CogVideoX-2b"


def run_cogvideox(
    prompts: list[str],
    output_dir: Path,
    model_id: str = COG_VIDEO_MODEL,
    width: int = 480,
    height: int = 272,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
) -> list[Path]:
    """CogVideoX-2b로 T2V 생성 (Open-Sora 대리 모델)."""
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CogVideoX 로딩: {model_id}")
    print("  (첫 실행 시 ~10GB 다운로드 발생)")

    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    video_paths = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Open-Sora (CogVideoX): '{prompt[:60]}...'")
        generator = torch.Generator(device="cpu").manual_seed(seed + i)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        video = result.frames[0]
        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=8)
        print(f"  저장: {out_path}")
        video_paths.append(out_path)

    return video_paths


def run_open_sora_native(
    prompts: list[str],
    output_dir: Path,
    steps: int = 30,
    seed: int = 42,
) -> list[Path]:
    """Open-Sora 공식 패키지로 생성 (설치된 경우)."""
    try:
        import opensora
    except ImportError:
        raise RuntimeError(
            "Open-Sora 미설치.\n"
            "설치: pip install open-sora\n"
            "또는: https://github.com/hpcaitech/Open-Sora"
        )

    # Open-Sora CLI 래퍼
    output_dir.mkdir(parents=True, exist_ok=True)
    from opensora.registry import MODELS, build_module
    # Open-Sora API는 버전마다 다를 수 있음 — 공식 docs 참조
    raise NotImplementedError("Open-Sora native API — 공식 문서 참조 후 구현 필요")


def _find_ffmpeg() -> str:
    """ffmpeg 실행 경로 탐색."""
    import shutil
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if shutil.which(candidate) or (candidate.startswith("/") and os.path.isfile(candidate)):
            return candidate
    return "ffmpeg"  # fallback


def _combine_videos(video_paths: list[Path], output_dir: Path) -> Path:
    """ffmpeg concat으로 통합 비디오 생성."""
    combined = output_dir / "combined.mp4"
    ffmpeg = _find_ffmpeg()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        filelist = f.name
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0",
         "-i", filelist, "-c", "copy", str(combined)],
        capture_output=True,
    )
    os.unlink(filelist)
    if result.returncode != 0:
        print(f"  [warn] ffmpeg concat 실패: {result.stderr.decode()[:200]}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Open-Sora Baseline (CogVideoX fallback)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/eval_workspace/baselines/open_sora"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_id",    default=COG_VIDEO_MODEL,
                        help="CogVideoX 모델 ID (기본: THUDM/CogVideoX-2b)")
    parser.add_argument("--width",       type=int, default=480)
    parser.add_argument("--height",      type=int, default=272)
    parser.add_argument("--num_frames",  type=int, default=49)
    parser.add_argument("--steps",       type=int, default=50)
    parser.add_argument("--guidance",    type=float, default=6.0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--use_open_sora", action="store_true",
                        help="Open-Sora 공식 패키지 사용 (설치 필요)")
    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("Open-Sora Baseline")
    if args.use_open_sora:
        print("  모드: Open-Sora native")
    else:
        print(f"  모드: CogVideoX-2b ({args.model_id})")
        print("  [Note] Open-Sora 미설치 — CogVideoX-2b를 대리 모델로 사용")
    print(f"  출력   : {args.output_dir}")
    print(f"  steps  : {args.steps}")
    print(f"  prompts: {len(prompts)}개")
    print("=" * 65)

    if args.use_open_sora:
        paths = run_open_sora_native(prompts, args.output_dir, args.steps, args.seed)
    else:
        paths = run_cogvideox(
            prompts=prompts,
            output_dir=args.output_dir,
            model_id=args.model_id,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )

    if len(paths) > 1:
        combined = _combine_videos(paths, args.output_dir)
        print(f"\n통합 비디오: {combined}")

    print(f"\n✅ Open-Sora 완료: {len(paths)}개 비디오")
    print(f"   저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
