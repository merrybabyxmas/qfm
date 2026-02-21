#!/usr/bin/env python
"""
QSFM Scaling Law Benchmark  (Table 2)
======================================
K 샷 수에 따른 VRAM / Latency 비교:
  - QSFM 양자 모듈 (D_sys = K × d_latent = 64 고정)
  - 고전 Self-Attention baseline (K 토큰 시퀀스)

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    python scripts/benchmark_scaling.py [--n_runs 5] [--output_csv outputs/scaling_benchmark.csv]

Table 2 논문 아이디어:
  QSFM : K가 4→32 로 늘어도 VRAM/time ≈ 상수 (D_sys 고정)
  Attn : K가 증가할수록 VRAM/time ∝ K²  (attention cost)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.qsfm import QSFMModule  # noqa: E402

# ── 설정 ─────────────────────────────────────────────────────────────────
LATENT_DIM = 128   # LTX-Video 잠재 채널 수

# K 프리셋: (K, n_idx_qubits, n_latent_qubits)
# D_sys = 2^n_idx * 2^n_latent = 64  고정
K_PRESETS = [
    ( 4, 2, 4),   # d_latent=16  (best quality)
    ( 8, 3, 3),   # d_latent=8
    (16, 4, 2),   # d_latent=4
    (32, 5, 1),   # d_latent=2
]


# ── 고전 Attention Baseline ───────────────────────────────────────────────
class ClassicalAttentionBaseline(nn.Module):
    """K 샷을 K-토큰 시퀀스로 보고 Self-Attention 적용 (고전 DiT baseline)."""

    def __init__(self, latent_dim: int, K: int):
        super().__init__()
        # K에 맞게 head 수 조정 (latent_dim=128, K≤128)
        n_heads = min(8, latent_dim // 16)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, K, latent_dim)
        out, _ = self.attn(x, x, x)
        return self.proj(out)


# ── 측정 함수 ─────────────────────────────────────────────────────────────
def measure_qsfm(
    qsfm: QSFMModule,
    K: int,
    device: torch.device,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> tuple[float, float]:
    """QSFM forward pass 시간(ms)과 peak VRAM(MB) 반환."""
    qsfm = qsfm.to(device)
    qsfm.eval()

    dummy_shots = [torch.randn(1, LATENT_DIM, device=device) for _ in range(K)]
    t_qsfm = torch.tensor([0.5], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            qsfm.training_forward(dummy_shots, t_qsfm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            qsfm.training_forward(dummy_shots, t_qsfm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else 0.0
    )
    return avg_ms, peak_mb


def measure_attention(
    K: int,
    device: torch.device,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> tuple[float, float]:
    """Classical Attention baseline 시간(ms)과 peak VRAM(MB) 반환."""
    model = ClassicalAttentionBaseline(LATENT_DIM, K).to(device)
    model.eval()
    dummy = torch.randn(1, K, LATENT_DIM, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else 0.0
    )
    return avg_ms, peak_mb


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QSFM Scaling Law Benchmark")
    parser.add_argument("--n_runs",     type=int, default=5,
                        help="Number of benchmark runs (default 5)")
    parser.add_argument("--n_warmup",   type=int, default=2,
                        help="Number of warmup runs (default 2)")
    parser.add_argument("--output_csv", type=Path,
                        default=Path("outputs/scaling_benchmark.csv"),
                        help="CSV output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("QSFM Scaling Law Benchmark  ←  논문 Table 2")
    print(f"Device : {device}  |  n_runs={args.n_runs}  |  LATENT_DIM={LATENT_DIM}")
    print("=" * 80)
    print()

    col_w = [4, 6, 9, 6, 15, 14, 15, 14, 9]
    header = (
        f"{'K':>{col_w[0]}} | "
        f"{'n_idx':>{col_w[1]}} | "
        f"{'n_latent':>{col_w[2]}} | "
        f"{'D_sys':>{col_w[3]}} | "
        f"{'QSFM(ms)':>{col_w[4]}} | "
        f"{'QSFM VRAM':>{col_w[5]}} | "
        f"{'Attn(ms)':>{col_w[6]}} | "
        f"{'Attn VRAM':>{col_w[7]}} | "
        f"{'Speedup':>{col_w[8]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    rows = []
    for K, n_idx, n_latent in K_PRESETS:
        D_sys = 2**n_idx * 2**n_latent

        # ── QSFM ─────────────────────────────────────────────
        try:
            qsfm = QSFMModule(
                latent_dim=LATENT_DIM,
                n_idx_qubits=n_idx,
                n_latent_qubits=n_latent,
                time_embed_dim=32,
                n_pqc_layers=1,
                n_ancilla_qubits=2,
            )
            qsfm_ms, qsfm_mb = measure_qsfm(
                qsfm, K, device, args.n_warmup, args.n_runs
            )
        except Exception as e:
            print(f"  [QSFM K={K}] 오류: {e}")
            qsfm_ms, qsfm_mb = float("nan"), float("nan")

        # ── Classical Attention ───────────────────────────────
        try:
            attn_ms, attn_mb = measure_attention(K, device, args.n_warmup, args.n_runs)
        except Exception as e:
            print(f"  [Attn K={K}] 오류: {e}")
            attn_ms, attn_mb = float("nan"), float("nan")

        speedup = (
            attn_ms / qsfm_ms
            if not (math.isnan(qsfm_ms) or math.isnan(attn_ms) or qsfm_ms == 0)
            else float("nan")
        )

        row_str = (
            f"{K:>{col_w[0]}} | "
            f"{n_idx:>{col_w[1]}} | "
            f"{n_latent:>{col_w[2]}} | "
            f"{D_sys:>{col_w[3]}} | "
            f"{qsfm_ms:>{col_w[4]}.2f} ms | "
            f"{qsfm_mb:>{col_w[5]-3}.1f} MB | "
            f"{attn_ms:>{col_w[6]}.2f} ms | "
            f"{attn_mb:>{col_w[7]-3}.1f} MB | "
            f"{speedup:>{col_w[8]}.2f}x"
        )
        print(row_str)
        rows.append((K, n_idx, n_latent, D_sys, qsfm_ms, qsfm_mb, attn_ms, attn_mb, speedup))

    print(sep)
    print()
    print("📌 핵심: QSFM VRAM / Time ≈ O(1) w.r.t. K  (D_sys=64 고정)")
    print("         Classical Attention : VRAM / Time ∝ K  (시퀀스 길이 증가)")
    print()

    # ── CSV 저장 ─────────────────────────────────────────────────────────
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write("K,n_idx_qubits,n_latent_qubits,D_sys,"
                "qsfm_time_ms,qsfm_vram_mb,"
                "attn_time_ms,attn_vram_mb,speedup\n")
        for row in rows:
            f.write(",".join(
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in row
            ) + "\n")

    print(f"✅ CSV 저장 완료: {args.output_csv}")


if __name__ == "__main__":
    main()
