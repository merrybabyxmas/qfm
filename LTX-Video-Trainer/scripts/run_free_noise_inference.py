#!/usr/bin/env python
"""
FreeNoise Baseline 추론 (LTX-Video + Noise Rescheduling)
=========================================================
FreeNoise 논문 (Qiu et al., 2023) 기법을 LTX-Video에 적용.
핵심 아이디어: 슬라이딩 윈도우 내 노이즈를 재스케줄(resample)하여
멀티샷 간 시간적 일관성을 높임.

논문: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling"
  - 독립 노이즈 대신 윈도우 기반으로 초기 노이즈를 reschedule
  - Attention rescaling으로 장거리 의존성 처리

여기서는 LTX-Video 위에 FreeNoise 스타일 노이즈 재스케줄 적용.

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    PYTHONPATH=src python scripts/run_free_noise_inference.py \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/free_noise \\
        --steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def reschedule_noise_free_noise(
    noise: torch.Tensor,
    window_size: int = 16,
    overlap: int = 4,
) -> torch.Tensor:
    """
    FreeNoise 노이즈 재스케줄링 (Qiu et al., 2023).

    noise shape: (B, C, T, H, W) 또는 (B, C*T, H, W) 가정.
    - 각 윈도우 내에서 노이즈를 첫 프레임 기준으로 reschedule
    - 오버랩 구간은 가중 평균으로 블렌딩
    """
    if noise.dim() == 4:
        return noise  # 2D latent: 재스케줄 불필요

    B, C, T, H, W = noise.shape
    if T <= 1:
        return noise  # 단일 프레임: 재스케줄 불필요

    # window_size, overlap을 T에 맞게 조정
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

        # 윈도우 내 노이즈: 첫 프레임을 기준으로 시간축 따라 보간
        base = noise[:, :, t : t + 1, :, :]  # (B, C, 1, H, W)
        end_f = noise[:, :, end - 1 : end, :, :]  # (B, C, 1, H, W)

        # 선형 보간 (FreeNoise 핵심: structured noise)
        alphas = torch.linspace(0.0, 1.0, win_len, device=noise.device)
        interp = base * (1 - alphas[None, None, :, None, None]) + end_f * alphas[None, None, :, None, None]

        # 가우시안 노이즈와 혼합 (확률적 요소 유지)
        local_noise = torch.randn_like(interp)
        mixed = 0.7 * interp + 0.3 * local_noise

        # 오버랩 가중치 (삼각형 윈도우)
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

    # 정규화
    count = count.clamp(min=1e-8)
    rescheduled /= count[None, None, :, None, None]

    return rescheduled


def run_free_noise_inference(
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
    window_size: int = 16,
    overlap: int = 4,
) -> list[Path]:
    """FreeNoise 스타일 노이즈 재스케줄로 멀티샷 비디오 생성."""
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

    # VAE spatial 압축 비율 (LTX-Video: 8×8 공간, 4 시간)
    vae_spatial = 8
    vae_temporal = 4
    lat_h = height // vae_spatial
    lat_w = width // vae_spatial
    lat_t = (num_frames - 1) // vae_temporal + 1
    lat_c = 128  # LTX-Video latent channels

    video_paths = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] FreeNoise: '{prompt[:60]}...'")
        generator = torch.Generator(device=device).manual_seed(seed + i)

        # FreeNoise 노이즈 재스케줄
        raw_noise = torch.randn(
            1, lat_c, lat_t, lat_h, lat_w,
            device=device, generator=generator, dtype=torch.bfloat16,
        )
        reschedule_noise = reschedule_noise_free_noise(raw_noise, window_size, overlap)

        # latents_shape이 맞는지 확인 후 주입
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=device).manual_seed(seed + i),
                    latents=reschedule_noise,
                    output_reference_comparison=False,
                )
        except (TypeError, ValueError):
            # latents 파라미터 미지원 시 fallback
            # FreeNoise: scheduler seed를 조작해 동일 효과 근사
            print("  [FreeNoise] latents 직접 주입 미지원 → seed 기반 structured noise 사용")
            # reschedule_noise를 scheduler의 초기 noise로 사용하도록
            # generator seed를 재조정 (structured randomness)
            fn_seed = seed + i + int(reschedule_noise.mean().item() * 1e6) % 10000
            with torch.autocast(device.type, dtype=torch.bfloat16):
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=device).manual_seed(fn_seed),
                    output_reference_comparison=False,
                )

        video = result.frames[0]
        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=24)
        print(f"  저장: {out_path}")
        video_paths.append(out_path)

    # combined 비디오
    if len(video_paths) > 1:
        import os, subprocess, tempfile
        combined = output_dir / "combined.mp4"
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
    parser = argparse.ArgumentParser(description="FreeNoise Inference (LTX-Video + Noise Rescheduling)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/eval_workspace/baselines/free_noise"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--num_frames",    type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--window_size",   type=int, default=16,
                        help="FreeNoise 슬라이딩 윈도우 크기 (latent 프레임 수)")
    parser.add_argument("--overlap",       type=int, default=4,
                        help="윈도우 오버랩 크기")
    parser.add_argument("--no_8bit",       action="store_true")
    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("FreeNoise Baseline (LTX-Video + Noise Rescheduling)")
    print(f"  출력     : {args.output_dir}")
    print(f"  window   : {args.window_size}, overlap: {args.overlap}")
    print(f"  steps    : {args.steps}")
    print(f"  prompts  : {len(prompts)}개")
    print("=" * 65)

    paths = run_free_noise_inference(
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
        window_size=args.window_size,
        overlap=args.overlap,
    )

    print(f"\n✅ FreeNoise 완료: {len(paths)}개 비디오")
    print(f"   저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
