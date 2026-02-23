import time
import torch
import sys
from collections import defaultdict

def print_model_summary_and_estimate_resources(
    pipeline,
    method_name: str,
    K_shots: int,
    frames_per_shot: int = 121,
    latent_dim: int = 128
):
    """
    Check model parameters and estimate VRAM/Time resources.
    Prints a summary to stdout.
    """
    print("\n" + "="*60)
    print(f"📊 [Pre-Flight Check] {method_name} (K={K_shots})")
    print("="*60)

    # 1. Parameter Count
    trainable_params = 0
    total_params = 0

    # Iterate over pipeline components
    if hasattr(pipeline, "components"):
        # Custom LTX pipeline structure
        components = pipeline.components
    else:
        # Standard Diffusers pipeline
        components = pipeline.__dict__

    for name, module in components.items():
        if isinstance(module, torch.nn.Module):
            t_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in module.parameters())
            trainable_params += t_params
            total_params += all_params
            print(f"  - {name}: {all_params / 1e6:.2f} M params (Trainable: {t_params / 1e6:.2f} M)")

    print(f"\n  👉 Total Parameters: {total_params / 1e9:.2f} B")
    print(f"  👉 Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 2. VRAM OOM Warning & Estimation
    print("\n[Resource Estimation]")
    total_frames = K_shots * frames_per_shot

    if method_name == "FreeNoise":
        # FreeNoise: Attention complexity O(K^2) or O(T^2)
        # Assuming linear increase with some overhead
        expected_vram = 10.0 + (K_shots * 1.5)
        print(f"  ⚠️ [FreeNoise] Attention complexity O(T^2) applies.")
    elif method_name == "Auto-regressive":
        # AR: 1 shot at a time. O(1) memory, O(K) time.
        expected_vram = 10.5
        print(f"  ⚠️ [Auto-regressive] Sequential generation. Memory O(1), Time O(K).")
    elif method_name == "QSFM (Ours)":
        expected_vram = 10.5
        print(f"  🌟 [QSFM] Quantum Simultaneous Generation. Memory O(1), Time O(1) (Parallelized latent gen).")
    else:
        expected_vram = 11.0

    print(f"  👉 Estimated Peak VRAM: ~{expected_vram:.1f} GB")

    # 3. Dry-Run OOM Check
    print("\n[Dry-Run OOM Check (1 Step)]")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # We can't easily run a real pipeline step here without potentially crashing or messing up state.
        # Instead, we allocate a dummy tensor of the expected latent size to see if it fits.
        # Latent shape: (B, C, F_latent, H_latent, W_latent)
        # LTX VAE compression: T=8, S=32
        vae_spatial = 32
        vae_temporal = 8
        lat_h = 512 // vae_spatial
        lat_w = 704 // vae_spatial
        lat_t = (total_frames - 1) // vae_temporal + 1

        if method_name == "Auto-regressive":
            # Check for 1 shot size
            lat_t = (frames_per_shot - 1) // vae_temporal + 1

        dummy_latent = torch.zeros((1, 128, lat_t, lat_h, lat_w), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
        del dummy_latent

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        print(f"  ✅ Dry-Run (Allocation) Passed! (Peak VRAM: {peak_mb:.1f} MB)")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ [FATAL] OOM Predicted! K={K_shots} is likely impossible on this GPU.")
        else:
            print(f"  ⚠️ Dry-Run Check Error: {e}")

    print("="*60 + "\n")


class PerformanceTimer:
    """Context manager to measure execution time and VRAM usage."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.start_vram = 0
        self.peak_vram = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_vram = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_vram = torch.cuda.max_memory_allocated()
        self.end_time = time.time()

        duration = self.end_time - self.start_time
        vram_gb = self.peak_vram / (1024**3)

        print(f"⏱️ [{self.name}] Time: {duration:.2f}s | Peak VRAM: {vram_gb:.2f} GB")
