#!/usr/bin/env python
"""
Benchmark SOTA Scaling & Quality Evaluation (Table 1 & Table 2)
===============================================================
1. 지정된 프롬프트로 AR, FreeNoise, QSFM 모델의 비디오를 생성합니다.
2. 각 K(샷 수)별로 VRAM과 생성 속도(Denoising Time)를 측정하여 Table 2를 만듭니다.
3. 생성이 끝난 직후, 각 출력 폴더에 대해 `eval_multi_shot_metrics.py`를 호출하여
   비디오 품질 및 컷편집 일관성 지표(CLIPSIM, DINO)를 자동 측정합니다 (Table 1).
"""

import subprocess
import re
import csv
import sys
import json
from pathlib import Path

# ==========================================
# ⚙️ 1. 실험 세팅 및 프롬프트 정의
# ==========================================
K_VALUES = [4, 8, 16] # 시간 절약을 위해 32는 필요시 추가하세요.

METHODS = {
    "Auto-regressive": "scripts/run_autoregressive_inference.py",
    "FreeNoise": "scripts/run_free_noise_inference.py",
    "QSFM": "scripts/run_qsfm_inference.py",
}

# 🌟 순정 베이스 모델 (뼈대)
BASE_MODEL = "LTXV_2B_0.9.6_DEV"

BASELINE_LORA_DIR = Path("/home/dongwoo43/qfm/LTX-Video-Trainer/outputs/ltxv_lora/checkpoints")
QSFM_LORA_DIR = Path("/home/dongwoo43/qfm/LTX-Video-Trainer/outputs/qsfm_lora/checkpoints")
OUTPUT_CSV = "outputs/benchmark_results.csv"

# 🐰 사용자가 지정한 4개의 표준 프롬프트
STANDARD_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]

def get_latest_lora(checkpoint_dir: Path):
    """폴더 내에서 가장 마지막 스텝의 .safetensors 파일을 자동으로 찾습니다."""
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.glob("*.safetensors"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]) if "_" in x.stem else 0)
    return ckpts[-1]

def parse_output(output: str):
    """표준 출력에서 VRAM과 생성 속도를 파싱합니다."""
    metrics = {
        "Peak VRAM (GB)": 0.0,
        "Denoising Time (s)": 0.0,
        "Total Time (s)": 0.0,
        "OOM": False
    }

    if "CUDA out of memory" in output or "OOM Predicted" in output or "out of memory" in output.lower():
        metrics["OOM"] = True
        return metrics

    vram_matches = re.findall(r"Peak VRAM: ([\d\.]+) GB", output)
    if vram_matches:
        metrics["Peak VRAM (GB)"] = max(float(v) for v in vram_matches)

    dt_match = re.search(r"\[METRICS\] Denoising Time per Shot: ([\d\.]+)", output)
    if dt_match:
        metrics["Denoising Time (s)"] = float(dt_match.group(1))

    tt_match = re.search(r"\[METRICS\] Total Generation Time: ([\d\.]+)", output)
    if tt_match:
        metrics["Total Time (s)"] = float(tt_match.group(1))

    return metrics

def run_benchmark():
    results = [] 

    baseline_lora_path = get_latest_lora(BASELINE_LORA_DIR)
    qsfm_lora_path = get_latest_lora(QSFM_LORA_DIR)

    print("=" * 80)
    print("🚀 SOTA Macro-Benchmark & Quality Evaluation Pipeline")
    print(f" - Base Model: {BASE_MODEL}")
    print(f" - Baseline LoRA: {baseline_lora_path}")
    print(f" - QSFM LoRA: {qsfm_lora_path}")
    print("=" * 80)
    print(f"{'Method':<20} | {'K':<4} | {'VRAM (GB)':<10} | {'Time/Shot (s)':<15} | {'Status'}")
    print("-" * 70)

    for method_name, script_path in METHODS.items():
        for K in K_VALUES:
            output_dir = Path(f"eval_workspace/benchmark/{method_name.lower().replace(' ', '_')}_k{K}")
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, script_path]
            cmd.extend(["--output_dir", str(output_dir)])
            cmd.extend(["--seed", "42"])
            cmd.extend(["--steps", "30"]) # 논문용 품질을 위해 30스텝 권장

            cmd.extend(["--model_source", BASE_MODEL])

            if method_name in ["Auto-regressive", "FreeNoise"] and baseline_lora_path:
                cmd.extend(["--lora_weights_path", str(baseline_lora_path)])
            elif method_name == "QSFM" and qsfm_lora_path:
                cmd.extend(["--lora_weights_path", str(qsfm_lora_path)])

            # ==========================================
            # 📝 2. 각 메소드에 맞는 프롬프트 분배
            # ==========================================
            prompts_file = output_dir / "prompts.json"
            
            if method_name == "Auto-regressive":
                # AR은 K개에 맞춰 프롬프트를 반복/할당
                cycled_prompts = [STANDARD_PROMPTS[i % len(STANDARD_PROMPTS)] for i in range(K)]
                prompts_data = {"standard": [{"prompt": p} for p in cycled_prompts]}
            else:
                # FreeNoise와 QSFM은 단일 프롬프트에서 K샷을 뽑거나, 내부적으로 K를 조절함
                prompts_data = {"standard": [{"prompt": STANDARD_PROMPTS[0]}]}
                cmd.extend(["--num_shots", str(K)])
                
            prompts_file.write_text(json.dumps(prompts_data))
            cmd.extend(["--prompts_json", str(prompts_file)])

            # ==========================================
            # 🏃 3. 비디오 생성 실행 (하드웨어 지표 측정)
            # ==========================================
            status = "Unknown"
            metrics = parse_output("")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
                metrics = parse_output(result.stdout)

                if result.returncode != 0 and not metrics["OOM"]:
                    status = "Error"
                    print(f"Error in {method_name} K={K}:\n{result.stderr[-300:]}")
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

            vram_display = f"{metrics['Peak VRAM (GB)']:.2f}" if isinstance(metrics['Peak VRAM (GB)'], float) else str(metrics['Peak VRAM (GB)'])
            time_display = f"{metrics['Denoising Time (s)']:.2f}" if isinstance(metrics['Denoising Time (s)'], float) else str(metrics['Denoising Time (s)'])
            print(f"{method_name:<20} | {K:<4} | {vram_display:<10} | {time_display:<15} | {status}")

            results.append({
                "Method": method_name, "K": K, "VRAM": metrics["Peak VRAM (GB)"],
                "Time_per_Shot": metrics["Denoising Time (s)"], "Total_Time": metrics["Total Time (s)"]
            })

            # ==========================================
            # 📊 4. 품질 및 성능 지표 즉시 자동 평가 (Table 1 용)
            # ==========================================
            if status == "Success":
                print(f"  └─> [Evaluating] {method_name} K={K} 비디오 품질 분석 중...")
                # 🚀 수정 완료: --shots_dir 와 --prompts_json 적용
                eval_cmd = [
                    sys.executable, "scripts/eval_multi_shot_metrics.py", 
                    "--shots_dir", str(output_dir),
                    "--prompts_json", str(prompts_file)
                ]
                try:
                    subprocess.run(eval_cmd, check=False)
                except Exception as eval_e:
                    print(f"      평가 스크립트 실행 오류: {eval_e}")

    # ==========================================
    # 💾 5. 하드웨어 지표 결과 저장 (Table 2 용)
    # ==========================================
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "K", "VRAM", "Time_per_Shot", "Total_Time"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "="*80)
    print("📊 Table 2. Scaling Law (Hardware Efficiency)")
    print("="*80)
    header = f"{'Method':<20} | {'K=4 VRAM/Time':<20} | {'K=8 VRAM/Time':<20} | {'K=16 VRAM/Time':<20}"
    print(header)
    print("-" * len(header))

    method_data = {m: {} for m in METHODS.keys()}
    for row in results:
        m, k = row["Method"], row["K"]
        vram = f"{row['VRAM']:.1f}G" if isinstance(row['VRAM'], float) else row['VRAM']
        time = f"{row['Time_per_Shot']:.1f}s" if isinstance(row['Time_per_Shot'], float) else str(row['Time_per_Shot'])
        method_data[m][k] = f"{vram} / {time}"

    for m in METHODS.keys():
        row_str = f"{m:<20} | "
        for k in K_VALUES:
            val = method_data[m].get(k, "N/A")
            row_str += f"{val:<20} | "
        print(row_str)

    print(f"\n✅ 완료되었습니다! 하드웨어 결과는 {OUTPUT_CSV} 에 저장되었으며, 품질 평가 결과는 위 콘솔 로그를 확인하세요.")

if __name__ == "__main__":
    run_benchmark()