#!/usr/bin/env python
"""
FreeNoise Baseline Inference (LTX-Video + Noise Rescheduling)
=========================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.profiling import print_model_summary_and_estimate_resources, PerformanceTimer

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

def reschedule_noise_free_noise(
    noise: torch.Tensor,
    window_size: int = 16,
    overlap: int = 4,
) -> torch.Tensor:
    if noise.dim() == 4:
        return noise

    B, C, T, H, W = noise.shape
    if T <= 1:
        return noise

    window_size = min(window_size, T)
    overlap = min(overlap, window_size - 1, T - 1)
    if overlap < 0:
        overlap = 0

    rescheduled = torch.zeros_like(noise)
    count = torch.zeros(T, device=noise.device)

    step = max(window_size - overlap, 1)

    t = 0
    while t < T:
        end = min(t + window_size, T)
        win_len = end - t
        if win_len <= 0:
            break

        base = noise[:, :, t : t + 1, :, :]
        end_f = noise[:, :, end - 1 : end, :, :]

        alphas = torch.linspace(0.0, 1.0, win_len, device=noise.device)
        interp = base * (1 - alphas[None, None, :, None, None]) + end_f * alphas[None, None, :, None, None]

        local_noise = torch.randn_like(interp)
        mixed = 0.7 * interp + 0.3 * local_noise

        weights = torch.ones(win_len, device=noise.device)
        eff_overlap = min(overlap, win_len)
        if t > 0 and eff_overlap > 0:
            weights[:eff_overlap] = torch.linspace(0.0, 1.0, eff_overlap, device=noise.device)
        if end < T and eff_overlap > 0:
            tail = min(eff_overlap, win_len)
            weights[-tail:] = torch.linspace(1.0, 0.0, tail, device=noise.device)

        rescheduled[:, :, t:end] += mixed * weights[None, None, :, None, None]
        count[t:end] += weights

        t += step

    count = count.clamp(min=1e-8)
    rescheduled /= count[None, None, :, None, None]

    return rescheduled

def split_video_frames(frames: list, num_shots: int) -> list[list]:
    total = len(frames)
    chunk_size = total // num_shots
    shots = []
    for i in range(num_shots):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_shots - 1 else total
        shots.append(frames[start:end])
    return shots

def run_free_noise_inference(
    prompts: list[str],
    output_dir: Path,
    model_source: str = "LTXV_2B_0.9.6_DEV",
    lora_weights_path: Path = None,
    width: int = 512,
    height: int = 320,
    frames_per_shot: int = 97,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    seed: int = 42,
    load_in_8bit: bool = True,
    window_size: int = 16,
    overlap: int = 4,
    num_shots: int = 1,
) -> list[Path]:
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

    total_frames = frames_per_shot * num_shots

    print_model_summary_and_estimate_resources(
        pipeline,
        method_name="FreeNoise",
        K_shots=num_shots,
        frames_per_shot=frames_per_shot
    )

    vae_spatial = 32 
    vae_temporal = 8
    lat_h = height // vae_spatial
    lat_w = width // vae_spatial
    lat_t = (total_frames - 1) // vae_temporal + 1
    lat_c = 128

    video_paths = []
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] FreeNoise (Long Video, K={num_shots}): '{prompt[:60]}...'")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        raw_noise = torch.randn(
            1, lat_c, lat_t, lat_h, lat_w,
            device=device, generator=generator, dtype=torch.bfloat16,
        )
        reschedule_noise = reschedule_noise_free_noise(raw_noise, window_size, overlap)

        with PerformanceTimer(f"FreeNoise Gen {i+1}") as timer:
            try:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_frames=total_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=device).manual_seed(seed + i),
                        latents=reschedule_noise,
                        output_reference_comparison=False,
                    )
            except (TypeError, ValueError):
                print("  [FreeNoise] fallback to scheduler seed manipulation")
                fn_seed = seed + i + int(reschedule_noise.mean().item() * 1e6) % 10000
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_frames=total_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=device).manual_seed(fn_seed),
                        output_reference_comparison=False,
                    )

        elapsed = timer.end_time - timer.start_time
        total_time += elapsed

        full_video = result.frames[0]
        shots = split_video_frames(full_video, num_shots)

        shot_paths = []
        for s_idx, shot_frames in enumerate(shots):
            # 🚨 🚀 핵심 수정 부분: test_000_shot_001.mp4 -> shot_001.mp4 
            out_path = output_dir / f"shot_{s_idx+1:03d}.mp4"
            export_to_video(shot_frames, str(out_path), fps=24)
            shot_paths.append(out_path)

        video_paths.extend(shot_paths)

    avg_per_shot = total_time / (len(prompts) * num_shots)
    print(f"\n[METRICS] Denoising Time per Shot: {avg_per_shot:.4f} s")
    print(f"[METRICS] Total Generation Time: {total_time:.4f} s")

    return video_paths

def main():
    parser = argparse.ArgumentParser(description="FreeNoise Inference")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/dongwoo43/qfm/eval_workspace/baselines/free_noise"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--lora_weights_path", type=Path, default=None)
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--frames_per_shot", type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--window_size",   type=int, default=16)
    parser.add_argument("--overlap",       type=int, default=4)
    parser.add_argument("--no_8bit",       action="store_true")
    parser.add_argument("--num_shots",     type=int, default=1)

    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    paths = run_free_noise_inference(
        prompts=prompts,
        output_dir=args.output_dir,
        model_source=args.model_source,
        lora_weights_path=args.lora_weights_path,
        width=args.width,
        height=args.height,
        frames_per_shot=args.frames_per_shot,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        load_in_8bit=not args.no_8bit,
        window_size=args.window_size,
        overlap=args.overlap,
        num_shots=args.num_shots,
    )

if __name__ == "__main__":
    main()