#!/usr/bin/env python
"""
LTX-Video Auto-regressive Baseline Inference
=========================================
Sequential generation where the last frame of the previous shot conditions the next shot.
Controlled Experiment Baseline for comparison with QSFM.

Comparisons:
  - Baseline 1 (This script): LTX-Video Auto-regressive
  - Baseline 2: LTX-Video + FreeNoise
  - Ours: LTX-Video + QSFM

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.profiling import print_model_summary_and_estimate_resources, PerformanceTimer

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def extract_last_frame(video_frames) -> Image.Image:
    """Extract the last frame from video frames as a PIL Image."""
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
    lora_weights_path: Path = None
) -> list[Path]:
    """
    LTX-Video Auto-regressive inference.
    """
    from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
    from ltxv_trainer.model_loader import LtxvModelVersion, load_ltxv_components

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Model: {model_source}")
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
    
    if lora_weights_path and lora_weights_path.exists():
        print(f"  [LoRA] 파인튜닝된 가중치를 덧씌웁니다: {lora_weights_path}")
        pipeline.load_lora_weights(str(lora_weights_path))

    # --- Profiling / Pre-flight Check ---
    print_model_summary_and_estimate_resources(
        pipeline,
        method_name="Auto-regressive",
        K_shots=len(prompts),
        frames_per_shot=num_frames
    )
    # ------------------------------------

    video_paths = []
    prev_last_frame: Image.Image | None = None

    total_generation_time = 0.0
    denoising_times = []

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

        with PerformanceTimer(f"Shot {i+1}") as timer:
            if prev_last_frame is not None:
                print(f"  Conditioning on previous shot (strength={conditioning_strength})")
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(
                        **common_kwargs,
                        image=prev_last_frame,
                        strength=conditioning_strength,
                        image_cond_noise_scale=image_cond_noise_scale,
                    )
            else:
                print("  First shot: Pure T2V")
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(**common_kwargs)

        # Accumulate time
        shot_time = timer.end_time - timer.start_time
        total_generation_time += shot_time
        denoising_times.append(shot_time)

        video = result.frames[0]
        prev_last_frame = extract_last_frame(video)

        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=24)
        print(f"  Saved: {out_path}")
        video_paths.append(out_path)

    # Print Metrics for Benchmark Parsing
    avg_denoise_time = sum(denoising_times) / len(denoising_times) if denoising_times else 0
    print(f"\n[METRICS] Denoising Time per Shot: {avg_denoise_time:.4f} s")
    print(f"[METRICS] Total Generation Time: {total_generation_time:.4f} s")

    # Combined Video
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
    print(f"\nCombined video: {combined}")


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
    parser.add_argument("--conditioning_strength", type=float, default=0.65)
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.15)
    parser.add_argument("--no_8bit",        action="store_true")
    parser.add_argument("--lora_weights_path", type=Path, default=None)
    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("LTX-Video Auto-regressive Baseline")
    print(f"  Output   : {args.output_dir}")
    print(f"  Steps    : {args.steps}")
    print(f"  Strength : {args.conditioning_strength}")
    print(f"  Prompts  : {len(prompts)} shots")
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
        lora_weights_path=args.lora_weights_path,
    )

    print(f"\n✅ Auto-regressive Complete: {len(paths)} videos")

if __name__ == "__main__":
    main()
