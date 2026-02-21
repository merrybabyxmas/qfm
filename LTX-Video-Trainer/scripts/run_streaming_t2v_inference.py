#!/usr/bin/env python
"""
StreamingT2V Baseline 추론 (Sliding-Window 조건부 생성)
======================================================
StreamingT2V (Henschel et al., 2024) 핵심 메커니즘:
  - 청크(chunk) 단위 생성 + 이전 청크 마지막 프레임을 조건으로 사용
  - Cross-frame attention으로 장거리 일관성 유지

공식 구현: https://github.com/Picsart-AI-Research/StreamingT2V
  → 별도 설치 필요 (pip install streaming-t2v)

여기서는 LTX-Video를 기반으로 동일 메커니즘 구현:
  1. 첫 번째 샷: 일반 T2V 생성
  2. 이후 샷: 이전 샷의 마지막 프레임을 condition으로 전달 (image2video)
  3. 멀티샷 간 시각적 일관성을 조건부 생성으로 확보

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    PYTHONPATH=src python scripts/run_streaming_t2v_inference.py \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/streaming_t2v \\
        --steps 30

    # 공식 StreamingT2V 설치 후
    PYTHONPATH=src python scripts/run_streaming_t2v_inference.py \\
        --use_streaming_t2v \\
        --output_dir /home/dongwoo43/qfm/eval_workspace/baselines/streaming_t2v
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
import numpy as np
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def frames_to_tensor(frames: list) -> torch.Tensor:
    """비디오 프레임 리스트 (PIL or np.ndarray) → (T, H, W, 3) uint8."""
    if not frames:
        return torch.zeros(1, 1, 1, 3, dtype=torch.uint8)
    arr = []
    for f in frames:
        if hasattr(f, "numpy"):
            arr.append(f)
        else:
            arr.append(torch.from_numpy(np.array(f)))
    return torch.stack(arr)


def get_last_frame_condition(frames: list | torch.Tensor) -> list | None:
    """이전 샷의 마지막 프레임 추출."""
    try:
        if isinstance(frames, torch.Tensor):
            last = frames[-1]
            return last.unsqueeze(0).tolist()
        # PIL 이미지 리스트
        return [frames[-1]]
    except Exception:
        return None


def run_streaming_generation(
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
    conditioning_strength: float = 0.6,
) -> list[Path]:
    """
    StreamingT2V 스타일: 이전 샷 마지막 프레임 조건부 순차 생성.

    conditioning_strength: 이전 프레임 영향 강도 (0=없음, 1=강함)
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
    prev_frames = None  # 이전 샷 마지막 프레임

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] StreamingT2V: '{prompt[:60]}...'")

        # 조건부 생성 시도 (이전 프레임 있을 때)
        if prev_frames is not None and conditioning_strength > 0:
            print(f"  이전 샷 조건부 생성 (strength={conditioning_strength})")
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
                        image=prev_frames[-1],  # 마지막 프레임 조건
                        conditioning_scale=conditioning_strength,
                        output_reference_comparison=False,
                    )
            except TypeError:
                # image 파라미터 미지원 → 표준 생성
                print("  [streaming] 조건부 image 파라미터 미지원 → 일반 생성")
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
                        output_reference_comparison=False,
                    )
        else:
            # 첫 번째 샷: 일반 T2V
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
                    output_reference_comparison=False,
                )

        video = result.frames[0]
        # 이전 프레임 업데이트 (마지막 프레임)
        prev_frames = video

        out_path = output_dir / f"shot_{i+1:03d}.mp4"
        export_to_video(video, str(out_path), fps=24)
        print(f"  저장: {out_path}")
        video_paths.append(out_path)

    # combined 비디오
    if len(video_paths) > 1:
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


def run_streaming_t2v_native(
    prompts: list[str],
    output_dir: Path,
    steps: int = 50,
    seed: int = 42,
) -> list[Path]:
    """StreamingT2V 공식 패키지 사용 (설치된 경우)."""
    try:
        import streaming_t2v  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "StreamingT2V 미설치.\n"
            "설치: https://github.com/Picsart-AI-Research/StreamingT2V"
        )
    raise NotImplementedError("StreamingT2V native API — 공식 문서 참조 후 구현 필요")


def main():
    parser = argparse.ArgumentParser(description="StreamingT2V Baseline")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/eval_workspace/baselines/streaming_t2v"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--num_frames",    type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--conditioning_strength", type=float, default=0.6,
                        help="이전 프레임 조건부 강도 (0~1)")
    parser.add_argument("--no_8bit",       action="store_true")
    parser.add_argument("--use_streaming_t2v", action="store_true",
                        help="공식 StreamingT2V 패키지 사용 (설치 필요)")
    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("StreamingT2V Baseline")
    if args.use_streaming_t2v:
        print("  모드: StreamingT2V native")
    else:
        print("  모드: LTX-Video + Sliding-Window 조건부 생성")
        print("  [Note] StreamingT2V 미설치 — 동일 메커니즘을 LTX-Video로 구현")
    print(f"  출력   : {args.output_dir}")
    print(f"  steps  : {args.steps}")
    print(f"  cond   : {args.conditioning_strength}")
    print(f"  prompts: {len(prompts)}개")
    print("=" * 65)

    if args.use_streaming_t2v:
        paths = run_streaming_t2v_native(prompts, args.output_dir, args.steps, args.seed)
    else:
        paths = run_streaming_generation(
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
        )

    print(f"\n✅ StreamingT2V 완료: {len(paths)}개 비디오")
    print(f"   저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
