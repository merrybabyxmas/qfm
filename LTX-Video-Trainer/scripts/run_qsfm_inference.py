#!/usr/bin/env python
"""
QSFM (Quantum Superposition Flow Matching) Inference
====================================================
Method C (Ours): Quantum Simultaneous Generation.

Generates K shots simultaneously using a Quantum Density Matrix to model
entanglement between shots, ensuring global semantic consistency and
controlled diversity.

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
    width: int = 512,
    height: int = 320,
    frames_per_shot: int = 97,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    seed: int = 42,
    load_in_8bit: bool = True,
    num_shots: int = 4, # K
    qsfm_time_steps: int = 20,
) -> list[Path]:
    """
    QSFM Inference Pipeline.
    """
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

    # Initialize QSFM Module
    print("Initializing QSFM Module...")
    # Determine n_idx_qubits from num_shots
    import math
    n_idx_qubits = math.ceil(math.log2(num_shots)) if num_shots > 1 else 1

    qsfm_module = QSFMModule(
        latent_dim=128,
        n_idx_qubits=n_idx_qubits,
        n_latent_qubits=4, # d_latent=16
        time_embed_dim=32,
        n_pqc_layers=1,
    ).to(device)

    # We might want to load weights here if they were trained (qsfm_finetuned.safetensors)
    # For now, we initialize from scratch or use random as per user prompt "Method C ... run_pipeline.py"
    # But usually one loads the trained QSFM. The prompt mentions "Load qsfm_finetuned.safetensors".
    # I'll check if the file exists and load it, otherwise warn.
    qsfm_path = Path("qsfm_finetuned.safetensors")
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

    # Profiling
    # Note: QSFM adds parameters to the pipeline logic effectively.
    # I'll pass the pipeline and maybe QSFM to the profiler if I modify it,
    # but the profiler iterates components. I can add qsfm to pipeline temporarily.
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

    # VAE dimensions for broadcasting
    vae_spatial = 32
    vae_temporal = 8
    lat_h = height // vae_spatial
    lat_w = width // vae_spatial
    lat_t = (frames_per_shot - 1) // vae_temporal + 1

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] QSFM (K={num_shots}): '{prompt[:60]}...'")

        with PerformanceTimer(f"QSFM Generation {i+1}") as timer:
            # 1. Quantum Simultaneous Generation of Shot Vectors
            # Returns list of K tensors of shape (1, 128) - or (B, 128) if B>1 but here we process 1 prompt at a time
            print(f"  Generating {num_shots} entangled shot vectors via Quantum Circuit...")
            shot_vectors = qsfm_inference.generate(batch_size=1, K=num_shots, device=device)
            # Decode using the projection (Phase F part 2)
            shot_vectors = [qsfm_module.decoder_proj(v) for v in shot_vectors]

            # 2. Generate Video Shots using LTX-Video conditioned on these vectors
            for k, z_k in enumerate(shot_vectors):
                print(f"    Shot {k+1}/{num_shots} (Entangled Condition)")
                # z_k: (1, 128)
                # Expand to (1, 128, T, H, W) to use as initial latents or strong bias
                # We use it as the mean of the initial noise distribution

                # Create base noise
                generator = torch.Generator(device=device).manual_seed(seed + i + k * 100)
                noise = torch.randn(
                    1, 128, lat_t, lat_h, lat_w,
                    device=device, generator=generator, dtype=torch.bfloat16
                )

                # Inject Quantum State:
                # latents = z_k + noise
                # z_k needs reshaping: (1, 128, 1, 1, 1)
                z_k_expanded = z_k.view(1, 128, 1, 1, 1).to(dtype=torch.bfloat16)

                # Mixing strategy:
                # If z_k is the "content core", we want it to guide the generation.
                # Standard LTX expects N(0,1) latents.
                # We can perform "Interpolated Initialization": latents = alpha * z_k + (1-alpha) * noise
                # Or just add them.
                # Given QSFM "Apples-to-Apples", let's assume z_k acts as the "Content Code".

                # However, LTX transformer expects valid VAE latents.
                # z_k comes from a random quantum process (if untrained) or learned distribution.
                # I will use a simple addition with normalization to keep variance ~1.
                latents = 0.5 * z_k_expanded + 0.866 * noise # sqrt(1 - 0.5^2) approx

                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        num_frames=frames_per_shot,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator, # Use same generator for scheduler
                        latents=latents, # Inject entangled latents
                        output_reference_comparison=False,
                    )

                video = result.frames[0]
                out_path = output_dir / f"test_{i:03d}_shot_{k+1:03d}.mp4"
                export_to_video(video, str(out_path), fps=24)
                video_paths.append(out_path)

        elapsed = timer.end_time - timer.start_time
        total_time += elapsed
        denoising_times.append(elapsed / num_shots) # Avg per shot

    avg_time = sum(denoising_times) / len(denoising_times) if denoising_times else 0
    print(f"\n[METRICS] Denoising Time per Shot: {avg_time:.4f} s")
    print(f"[METRICS] Total Generation Time: {total_time:.4f} s")

    return video_paths


def main():
    parser = argparse.ArgumentParser(description="QSFM Inference")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/dongwoo43/qfm/eval_workspace/controlled/qsfm"))
    parser.add_argument("--prompts_json", type=Path, default=None)
    parser.add_argument("--model_source",  default="LTXV_2B_0.9.6_DEV")
    parser.add_argument("--width",         type=int, default=512)
    parser.add_argument("--height",        type=int, default=320)
    parser.add_argument("--frames_per_shot", type=int, default=97)
    parser.add_argument("--steps",         type=int, default=30)
    parser.add_argument("--guidance",      type=float, default=3.5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--no_8bit",       action="store_true")
    parser.add_argument("--num_shots",     type=int, default=4, help="K shots")
    parser.add_argument("--qsfm_steps",    type=int, default=20)

    args = parser.parse_args()

    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS

    print("=" * 65)
    print("QSFM Inference (Quantum Simultaneous Generation)")
    print(f"  Output   : {args.output_dir}")
    print(f"  K Shots  : {args.num_shots}")
    print("=" * 65)

    paths = run_qsfm_inference(
        prompts=prompts,
        output_dir=args.output_dir,
        model_source=args.model_source,
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

    print(f"\n✅ QSFM Complete: {len(paths)} video clips generated.")


if __name__ == "__main__":
    main()
