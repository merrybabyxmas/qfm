#!/usr/bin/env python
"""
LTX-Video Auto-regressive Baseline 추론
=========================================
이전 샷의 마지막 프레임을 조건으로 다음 샷을 생성하는 순차 방식.
통제된 실험(Controlled Experiment)에서 QSFM과 동일한 Base Model로 비교.

비교 대상:
  - Baseline 1 (이 스크립트): LTX-Video Auto-regressive
  - Baseline 2: LTX-Video + FreeNoise  (run_free_noise_inference.py)
  - Ours: LTX-Video + QSFM

핵심 차이:
  - 이 방식: 샷 1 생성 → 마지막 프레임 추출 → 샷 2 조건부 생성 → 반복
  - QSFM: K개 샷을 양자 밀도 행렬로 동시 생성
  → 결과: Auto-regressive는 연속성은 높지만 다양성이 줄어들고
           후반 샷에서 주제(Subject)를 잃는 경향 발생

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    PYTHONPATH=src python scripts/run_autoregressive_inference.py \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/controlled/autoregressive \\
        --steps 20 \\
        --conditioning_strength 0.65
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def extract_last_frame(video_frames) -> Image.Image:
    """비디오 프레임 리스트에서 마지막 프레임을 PIL Image로 추출."""
    if not video_frames:
        return None
    last = video_frames[-1]
    if isinstance(last, Image.Image):
        return last
    # numpy array
    import numpy as np
    arr = np.array(last)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_autoregressive_inference(
    prompts: list[str],
    output_dir: Path,
    model_source: str = "LTXV_2B_0.9.6_DEV",
    width: int = 512,
    height: int = 320,
    num_frames: int = 97,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    seed: int = 42,
    load_in_8bit: bool = True,
    conditioning_strength: float = 0.65,
    image_cond_noise_scale: float = 0.15,
) -> list[Path]:
    """
    LTX-Video Auto-regressive: 이전 샷 마지막 프레임을 조건으로 순차 생성.

    Args:
        conditioning_strength: image2video 강도 (0=원본 이미지, 1=완전 노이즈)
            0.65 → 이전 샷의 맥락을 65% 반영하면서 새로운 장면 생성
        image_cond_noise_scale: 이미지 조건 노이즈 스케일
    """
    from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
    from ltxv_trainer.model_loader import LtxvModelVersion, load_ltxv_components

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"모델 로딩: {model_source}")
    try:
        version = LtxvModelVersion(model_source)
    except ValueError:
        version = None

    components = load_ltxv_components(
        model_source=version or model_source,
        load_text_encoder_in_8bit=load_in_8bit,
        transformer_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )

    pipeline = LTXConditionPipeline(
        scheduler=deepcopy(components.scheduler),
        vae=components.vae.to(device),
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        transformer=components.transformer.to(device),
    )
    pipeline.set_progress_bar_config(disable=False)

    video_paths = []
    prev_last_frame: Image.Image | None = None

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Auto-regressive: '{prompt[:60]}...'")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        common_kwargs = dict(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_reference_comparison=False,
        )

        if prev_last_frame is not None:
            # 이전 샷 마지막 프레임 조건부 생성
            print(f"  이전 샷 조건부 생성 (strength={conditioning_strength})")
            with torch.autocast(device.type, dtype=torch.bfloat16):
                result = pipeline(
                    **common_kwargs,
                    image=prev_last_frame,
                    strength=conditioning_strength,
                    image_cond_noise_scale=image_cond_noise_scale,
                )
        else:
            # 첫 번째 샷: 순수 T2V
            print("  첫 번째 샷: 순수 T2V 생성")
            with torch.autocast(device.type, dtype=torch.bfloat16):
                result = pipeline(**common_kwargs)

        video = result.frames[0]
        prev_last_frame = extract_last_frame(video)  # 다음 샷을 위한 조건 이미지

        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=24)
        print(f"  저장: {out_path}")
        video_paths.append(out_path)

    # Combined 비디오
    if len(video_paths) > 1:
        _combine_videos(video_paths, output_dir)

    return video_paths


def _combine_videos(video_paths: list[Path], output_dir: Path):
    import shutil
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    combined = output_dir / "combined.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        filelist = f.name
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0",
         "-i", filelist, "-c", "copy", str(combined)],
        capture_output=True,
    )
    os.unlink(filelist)
    print(f"\n통합 비디오: {combined}")


def main():
    parser = argparse.ArgumentParser(description="LTX-Video Auto-regressive Baseline")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/eval_workspace/controlled/autoregressive"))
    parser.add_argument("--prompts_json",   type=Path, default=None)
    parser.add_argument("--model_source",   default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--width",          type=int, default=512)
    parser.add_argument("--height",         type=int, default=320)
    parser.add_argument("--num_frames",     type=int, default=97)
    parser.add_argument("--steps",          type=int, default=30)
    parser.add_argument("--guidance",       type=float, default=3.5)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--conditioning_strength", type=float, default=0.65,
                        help="이미지 조건 강도: 낮을수록 이전 샷 맥락 강하게 반영")
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.15)
    parser.add_argument("--no_8bit",        action="store_true")
    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("LTX-Video Auto-regressive Baseline")
    print(f"  출력     : {args.output_dir}")
    print(f"  steps    : {args.steps}")
    print(f"  strength : {args.conditioning_strength} (낮을수록 이전 샷 영향 강함)")
    print(f"  prompts  : {len(prompts)}개")
    print("=" * 65)

    paths = run_autoregressive_inference(
        prompts=prompts,
        output_dir=args.output_dir,
        model_source=args.model_source,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        load_in_8bit=not args.no_8bit,
        conditioning_strength=args.conditioning_strength,
        image_cond_noise_scale=args.image_cond_noise_scale,
    )

    print(f"\n✅ Auto-regressive 완료: {len(paths)}개 비디오")
    print(f"   저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
