#!/usr/bin/env python
"""
Benchmark SOTA Scaling (Table 2 & Figure 1 Data)
================================================
Runs the three methods (Auto-regressive, FreeNoise, QSFM) with varying shot counts K.
Collects VRAM and Denoising Time metrics.
Generates the data for Table 2 and Figure 1 in the paper.

Usage:
    python scripts/benchmark_sota_scaling.py
"""

import subprocess
import re
import csv
import sys
import json
from pathlib import Path

# Configuration
K_VALUES = [4, 8, 16, 32]
METHODS = {
    "Auto-regressive": "scripts/run_autoregressive_inference.py",
    "FreeNoise": "scripts/run_free_noise_inference.py",
    "QSFM": "scripts/run_qsfm_inference.py",
}

OUTPUT_CSV = "outputs/benchmark_results.csv"

def parse_output(output: str):
    """Parses stdout for metrics."""
    metrics = {
        "Peak VRAM (GB)": 0.0,
        "Denoising Time (s)": 0.0,
        "Total Time (s)": 0.0,
        "OOM": False
    }

    # Check for OOM
    if "CUDA out of memory" in output or "OOM Predicted" in output or "out of memory" in output.lower():
        metrics["OOM"] = True
        return metrics

    # Parse Peak VRAM from PerformanceTimer
    # Pattern: ⏱️ [.*] Time: .* | Peak VRAM: 10.50 GB
    vram_matches = re.findall(r"Peak VRAM: ([\d\.]+) GB", output)
    if vram_matches:
        # Take the maximum VRAM observed across shots
        metrics["Peak VRAM (GB)"] = max(float(v) for v in vram_matches)

    # Parse Denoising Time per Shot
    # Pattern: [METRICS] Denoising Time per Shot: 8.0000 s
    dt_match = re.search(r"\[METRICS\] Denoising Time per Shot: ([\d\.]+)", output)
    if dt_match:
        metrics["Denoising Time (s)"] = float(dt_match.group(1))

    # Parse Total Time
    tt_match = re.search(r"\[METRICS\] Total Generation Time: ([\d\.]+)", output)
    if tt_match:
        metrics["Total Time (s)"] = float(tt_match.group(1))

    return metrics

def run_benchmark():
    results = [] # List of dicts

    print(f"{'Method':<20} | {'K':<4} | {'VRAM (GB)':<10} | {'Time/Shot (s)':<15} | {'Status'}")
    print("-" * 70)

    for method_name, script_path in METHODS.items():
        for K in K_VALUES:
            output_dir = Path(f"eval_workspace/benchmark/{method_name.lower().replace(' ', '_')}_k{K}")

            # Construct command
            cmd = [sys.executable, script_path]
            cmd.extend(["--output_dir", str(output_dir)])
            cmd.extend(["--seed", "42"])
            cmd.extend(["--steps", "20"])

            # Method specific arguments
            if method_name == "Auto-regressive":
                # Create prompt file for K shots
                prompts_data = {"standard": [{"prompt": "A rabbit in a forest."} for _ in range(K)]}
                output_dir.mkdir(parents=True, exist_ok=True)
                prompts_file = output_dir / "prompts.json"
                prompts_file.write_text(json.dumps(prompts_data))
                cmd.extend(["--prompts_json", str(prompts_file)])

            elif method_name == "FreeNoise":
                cmd.extend(["--num_shots", str(K)])

            elif method_name == "QSFM":
                cmd.extend(["--num_shots", str(K)])

            # Run
            status = "Unknown"
            metrics = parse_output("") # Default empty metrics

            try:
                # Timeout: 32 shots * 30s = ~16 mins. Set to 1500s.
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
                stdout = result.stdout
                stderr = result.stderr

                metrics = parse_output(stdout)

                if result.returncode != 0 and not metrics["OOM"]:
                    status = "Error"
                    # Print last few lines of error
                    err_lines = stderr.strip().split('\n')
                    print(f"Error in {method_name} K={K}: {err_lines[-3:] if err_lines else 'Unknown Error'}")
                elif metrics["OOM"]:
                    status = "OOM"
                    metrics["Peak VRAM (GB)"] = "OOM"
                    metrics["Denoising Time (s)"] = "-"
                else:
                    status = "Success"

            except subprocess.TimeoutExpired:
                status = "Timeout"
            except Exception as e:
                status = f"Error: {e}"

            # Print row
            vram_display = f"{metrics['Peak VRAM (GB)']:.2f}" if isinstance(metrics['Peak VRAM (GB)'], float) else str(metrics['Peak VRAM (GB)'])
            time_display = f"{metrics['Denoising Time (s)']:.2f}" if isinstance(metrics['Denoising Time (s)'], float) else str(metrics['Denoising Time (s)'])

            print(f"{method_name:<20} | {K:<4} | {vram_display:<10} | {time_display:<15} | {status}")

            results.append({
                "Method": method_name,
                "K": K,
                "VRAM": metrics["Peak VRAM (GB)"],
                "Time_per_Shot": metrics["Denoising Time (s)"],
                "Total_Time": metrics["Total Time (s)"]
            })

    # Save CSV
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "K", "VRAM", "Time_per_Shot", "Total_Time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")

    # Print Table 2 Format (Manual Pivot)
    print("\n" + "="*80)
    print("📊 Table 2. Scaling Law (Generated from Benchmark)")
    print("="*80)
    header = f"{'Method':<20} | {'K=4 VRAM/Time':<20} | {'K=8 VRAM/Time':<20} | {'K=16 VRAM/Time':<20} | {'K=32 VRAM/Time':<20}"
    print(header)
    print("-" * len(header))

    # Organize data by method
    method_data = {m: {} for m in METHODS.keys()}
    for row in results:
        m = row["Method"]
        k = row["K"]
        vram = f"{row['VRAM']:.1f}G" if isinstance(row['VRAM'], float) else row['VRAM']
        time = f"{row['Time_per_Shot']:.1f}s" if isinstance(row['Time_per_Shot'], float) else str(row['Time_per_Shot'])
        method_data[m][k] = f"{vram} / {time}"

    for m in METHODS.keys():
        row_str = f"{m:<20} | "
        for k in K_VALUES:
            val = method_data[m].get(k, "N/A")
            row_str += f"{val:<20} | "
        print(row_str)

if __name__ == "__main__":
    run_benchmark()
