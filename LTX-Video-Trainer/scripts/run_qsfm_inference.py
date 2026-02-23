#!/usr/bin/env python
"""
QSFM (Quantum Superposition Flow Matching) Inference
====================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.qsfm import QSFMModule, QSFMInference
from ltxv_trainer.profiling import print_model_summary_and_estimate_resources, PerformanceTimer

DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
]

NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

def run_qsfm_inference(
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
    num_shots: int = 4, 
    qsfm_time_steps: int = 20,
) -> list[Path]:
    from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
    from ltxv_trainer.model_loader import LtxvModelVersion, load_ltxv_components

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading LTX-Video: {model_source}")
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

    print("Initializing QSFM Module...")
    import math
    n_idx_qubits = math.ceil(math.log2(num_shots)) if num_shots > 1 else 1

    qsfm_module = QSFMModule(
        latent_dim=128,
        n_idx_qubits=n_idx_qubits,
        n_latent_qubits=4, 
        time_embed_dim=32,
        n_pqc_layers=1,
    ).to(device)

    # QSFM의 경우 lora 폴더 안에 같이 있다고 가정하거나 임의 초기화 사용
    qsfm_path = lora_weights_path.parent / "qsfm_finetuned.safetensors" if lora_weights_path else Path("qsfm_finetuned.safetensors")
    if qsfm_path.exists():
        print(f"Loading QSFM weights from {qsfm_path}")
        qsfm_module.load_state_dict(torch.load(qsfm_path, map_location=device))
    else:
        print("⚠️ QSFM weights not found. Using initialized weights (Random Entanglement).")

    qsfm_inference = QSFMInference(
        backward_channel=qsfm_module.backward_channel,
        amplitude_encoder=qsfm_module.amplitude_encoder,
        superposition_builder=qsfm_module.superposition_builder,
        n_inference_steps=qsfm_time_steps,
    )

    pipeline.qsfm = qsfm_module
    print_model_summary_and_estimate_resources(
        pipeline,
        method_name="QSFM (Ours)",
        K_shots=num_shots,
        frames_per_shot=frames_per_shot
    )

    video_paths = []
    total_time = 0.0
    denoising_times = []

    vae_spatial = 32
    vae_temporal = 8
    lat_h = height // vae_spatial
    lat_w = width // vae_spatial
    lat_t = (frames_per_shot - 1) // vae_temporal + 1

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] QSFM (K={num_shots}): '{prompt[:60]}...'")

        with PerformanceTimer(f"QSFM Generation {i+1}") as timer:
            print(f"  Generating {num_shots} entangled shot vectors via Quantum Circuit...")
            shot_vectors = qsfm_inference.generate(batch_size=1, K=num_shots, device=device)
            shot_vectors = [qsfm_module.decoder_proj(v) for v in shot_vectors]

            for k, z_k in enumerate(shot_vectors):
                print(f"    Shot {k+1}/{num_shots} (Entangled Condition)")
                generator = torch.Generator(device=device).manual_seed(seed + i + k * 100)
                noise = torch.randn(
                    1, 128, lat_t, lat_h, lat_w,
                    device=device, generator=generator, dtype=torch.bfloat16
                )

                z_k_expanded = z_k.view(1, 128, 1, 1, 1).to(dtype=torch.bfloat16)
                latents = 0.5 * z_k_expanded + 0.866 * noise 

                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_frames=frames_per_shot,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator, 
                        latents=latents, 
                        output_reference_comparison=False,
                    )

                video = result.frames[0]
                # 🚨 🚀 핵심 수정 부분: test_000_shot_001.mp4 -> shot_001.mp4 
                out_path = output_dir / f"shot_{k+1:03d}.mp4"
                export_to_video(video, str(out_path), fps=24)
                video_paths.append(out_path)

        elapsed = timer.end_time - timer.start_time
        total_time += elapsed
        denoising_times.append(elapsed / num_shots) 

    avg_time = sum(denoising_times) / len(denoising_times) if denoising_times else 0
    print(f"\n[METRICS] Denoising Time per Shot: {avg_time:.4f} s")
    print(f"[METRICS] Total Generation Time: {total_time:.4f} s")

    return video_paths

def main():
    parser = argparse.ArgumentParser(description="QSFM Inference")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/dongwoo43/qfm/eval_workspace/controlled/qsfm"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--lora_weights_path", type=Path, default=None)
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--frames_per_shot", type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--no_8bit",       action="store_true")
    parser.add_argument("--num_shots",     type=int, default=4)
    parser.add_argument("--qsfm_steps",    type=int, default=20)

    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    paths = run_qsfm_inference(
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
        num_shots=args.num_shots,
        qsfm_time_steps=args.qsfm_steps,
    )

if __name__ == "__main__":
    main()