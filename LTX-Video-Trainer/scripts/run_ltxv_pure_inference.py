#!/usr/bin/env python
"""
LTX-Video Pure Baseline 추론 (QSFM 없음)
==========================================
QSFM 없이 순정 LTX-Video 2B 모델로 표준 프롬프트에 대한 비디오를 생성.
논문 Table 1의 "LTX-Video (Base)" 항목을 위한 baseline.

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    python scripts/run_ltxv_pure_inference.py \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/ltx_video_pure \\
        --steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def run_inference(
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
    lora_path: str | None = None,
) -> list[Path]:
    """LTX-Video Pure로 프롬프트에 대한 비디오 생성."""
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

    # LoRA 가중치 로드 (있으면)
    if lora_path:
        print(f"LoRA 로드: {lora_path}")
        pipeline.load_lora_weights(lora_path)

    video_paths = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] '{prompt[:60]}...'")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = pipeline(
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

        video = result.frames[0]
        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=24)
        print(f"  저장: {out_path}")
        video_paths.append(out_path)

    # combined 비디오 생성
    if len(video_paths) > 1:
        combined = output_dir / "combined.mp4"
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for p in video_paths:
                f.write(f"file '{p}'\n")
            filelist = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", filelist, "-c", "copy", str(combined)],
            capture_output=True,
        )
        os.unlink(filelist)
        print(f"\n통합 비디오: {combined}")

    return video_paths


def main():
    parser = argparse.ArgumentParser(description="LTX-Video Pure Inference (Baseline)")
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("/home/dongwoo43/qfm/eval_workspace/baselines/ltx_video_pure"),
    )
    parser.add_argument("--prompts_json", type=Path, default=None,
                        help="prompts.json 경로 (없으면 기본 4개 사용)")
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--num_frames",    type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--lora_path",     type=str, default=None,
                        help="LoRA 가중치 경로 (QSFM-LoRA 비교용)")
    parser.add_argument("--no_8bit",       action="store_true",
                        help="8-bit 텍스트 인코더 비활성화")
    args = parser.parse_args()

    # 프롬프트 로드
    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
        print(f"prompts.json 로드: {len(prompts)}개")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"기본 프롬프트 사용: {len(prompts)}개")

    print("=" * 65)
    print("LTX-Video Pure Inference (QSFM 없음 — Baseline)")
    print(f"  model    : {args.model_source}")
    print(f"  출력     : {args.output_dir}")
    print(f"  해상도   : {args.width}×{args.height}×{args.num_frames}프레임")
    print(f"  steps    : {args.steps}")
    print(f"  prompts  : {len(prompts)}개")
    print("=" * 65)

    paths = run_inference(
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
        lora_path=args.lora_path,
    )

    print(f"\n✅ 생성 완료: {len(paths)}개 비디오")
    print(f"   저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
