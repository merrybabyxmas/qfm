#!/usr/bin/env python
"""
평가 워크스페이스 초기화 스크립트
===================================
논문 비교 실험을 위한 디렉토리 구조 생성 및 기존 QSFM 결과 정리.

생성되는 구조:
  eval_workspace/
    ground_truth/          ← 원본 비디오 (qsfm_data/scenes/ 에서 복사)
    baselines/
      ltx_video_pure/      ← LTX-Video (QSFM 없는 순정) 결과 저장 위치
      open_sora/           ← Open-Sora 결과 (외부 생성 후 배치)
      streaming_t2v/       ← StreamingT2V 결과 (외부 생성 후 배치)
      free_noise/          ← FreeNoise 결과 (외부 생성 후 배치)
    qsfm_outputs/
      qsfm_full/           ← QSFM Full (Hamiltonian ON) 현재 결과
      qsfm_no_hamiltonian/ ← QSFM Ablation A
      qsfm_gaussian/       ← QSFM Ablation B
    eval_results/          ← FVD, CLIPSIM 결과 CSV/JSON
    prompts.json           ← 표준 평가 프롬프트

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    python scripts/setup_eval_workspace.py [--workspace /home/dongwoo43/qfm/eval_workspace]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# 표준 평가 프롬프트 (K=4 기준, Table 1용)
STANDARD_PROMPTS = [
    {
        "id": "P001",
        "prompt": "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P002",
        "prompt": "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P003",
        "prompt": "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P004",
        "prompt": "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
        "source": "big_buck_bunny",
    },
]

# 확장 프롬프트 (K=8 이상 실험용)
EXTENDED_PROMPTS = [
    {
        "id": "P005",
        "prompt": "A cartoon rabbit runs away from three flying squirrels through a bright green forest.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P006",
        "prompt": "A rabbit and a squirrel interact playfully near a stream in a vibrant animated scene.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P007",
        "prompt": "Animated birds land on a resting rabbit in a peaceful forest setting.",
        "source": "big_buck_bunny",
    },
    {
        "id": "P008",
        "prompt": "A giant animated rabbit stretches and yawns in a sunlit forest meadow.",
        "source": "big_buck_bunny",
    },
]

README_BASELINES = """# Baselines 비교 모델 결과 배치 방법

각 SOTA 모델의 공식 코드로 동일한 프롬프트에 대해 비디오를 생성한 뒤
해당 서브폴더에 MP4 파일로 복사하세요.

## 권장 설정

- 해상도: 512×320 (QSFM과 동일)
- 프레임: 97 (≈ 4초, 24fps)
- 프롬프트: prompts.json의 P001~P004 (4-shot 실험)

## 모델별 코드베이스

- LTX-Video (순정):
    https://github.com/Lightricks/LTX-Video
    → QSFM 없이 동일 LTX-Video 모델로 추론만 수행

- Open-Sora:
    https://github.com/hpcaitech/Open-Sora

- StreamingT2V:
    https://github.com/Picsart-AI-Research/StreamingT2V

- FreeNoise:
    https://github.com/AILab-CVC/FreeNoise

## 파일명 규칙 (eval_metrics.py 호환)

생성 비디오 파일명은 자유롭게 지정 가능하나 combined는 제외:
  ltx_video_pure/
    shot_001.mp4
    shot_002.mp4
    shot_003.mp4
    shot_004.mp4
"""


def create_workspace(workspace: Path, qsfm_output_dir: Path, ground_truth_dir: Path) -> None:
    """워크스페이스 디렉토리 구조 생성."""

    subdirs = [
        "ground_truth",
        "baselines/ltx_video_pure",
        "baselines/open_sora",
        "baselines/streaming_t2v",
        "baselines/free_noise",
        "qsfm_outputs/qsfm_full",
        "qsfm_outputs/qsfm_no_hamiltonian",
        "qsfm_outputs/qsfm_gaussian",
        "eval_results",
    ]

    print(f"워크스페이스 생성: {workspace}")
    for sub in subdirs:
        (workspace / sub).mkdir(parents=True, exist_ok=True)
        print(f"  ✓  {sub}/")

    # ── 표준 프롬프트 JSON 저장 ──────────────────────────────────────
    prompts_all = STANDARD_PROMPTS + EXTENDED_PROMPTS
    json_path = workspace / "prompts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"standard": STANDARD_PROMPTS, "extended": EXTENDED_PROMPTS, "all": prompts_all},
            f, indent=2, ensure_ascii=False
        )
    print(f"\n  prompts.json 저장: {json_path}")

    # ── README 저장 ──────────────────────────────────────────────────
    readme = workspace / "baselines" / "README.md"
    readme.write_text(README_BASELINES, encoding="utf-8")
    print(f"  baselines/README.md 저장")

    # ── Ground Truth 복사 ────────────────────────────────────────────
    gt_dest = workspace / "ground_truth"
    if ground_truth_dir.exists():
        gt_videos = sorted([
            p for p in ground_truth_dir.glob("*.mp4")
        ])[:len(prompts_all)]
        copied = 0
        for p in gt_videos:
            dest = gt_dest / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
                copied += 1
        print(f"\n  ground_truth/: {copied}개 복사 (from {ground_truth_dir})")
    else:
        print(f"\n  ⚠ ground_truth 소스 없음: {ground_truth_dir}")
        print(f"    원본 비디오를 직접 {gt_dest} 에 복사하세요.")

    # ── QSFM Full 결과 복사 ──────────────────────────────────────────
    qsfm_dest = workspace / "qsfm_outputs" / "qsfm_full"
    qsfm_samples = qsfm_output_dir / "samples"
    if qsfm_samples.exists():
        # 가장 최신 step의 비디오 (combined 제외)
        all_steps = {}
        for p in qsfm_samples.glob("step_*_[0-9].mp4"):
            parts = p.stem.split("_")
            try:
                step = int(parts[1])
                all_steps.setdefault(step, []).append(p)
            except (IndexError, ValueError):
                pass

        if all_steps:
            latest_step = max(all_steps.keys())
            videos = sorted(all_steps[latest_step])[:len(STANDARD_PROMPTS)]
            copied = 0
            for i, p in enumerate(videos):
                dest = qsfm_dest / f"shot_{i+1:03d}.mp4"
                if not dest.exists():
                    shutil.copy2(p, dest)
                    copied += 1
            print(f"  qsfm_full/: step_{latest_step:06d}에서 {copied}개 복사")
        else:
            print(f"  ⚠ QSFM 결과 없음: {qsfm_samples}")
    else:
        print(f"  ⚠ QSFM samples 디렉토리 없음: {qsfm_samples}")
        print(f"    먼저 훈련을 완료하거나 수동으로 파일을 배치하세요.")

    print(f"\n{'='*60}")
    print("워크스페이스 준비 완료!")
    print(f"{'='*60}")
    print(f"경로: {workspace}")
    print()
    print("다음 단계:")
    print(f"  1. Ground Truth:  {gt_dest}")
    print(f"     → 원본 학습 데이터 비디오 배치")
    print(f"  2. Baselines:     {workspace}/baselines/*/")
    print(f"     → 각 SOTA 모델로 생성한 비디오 배치 (README.md 참고)")
    print(f"  3. 평가 실행:")
    print(f"""
conda activate afm
cd /home/dongwoo43/qfm/LTX-Video-Trainer

# Table 1: CLIPSIM + Temporal Consistency
python scripts/run_master_eval.py \\
    --workspace {workspace} \\
    --output_csv {workspace}/eval_results/table1.csv

# Table 2 (FVD)
python scripts/compute_fvd.py \\
    --workspace {workspace} \\
    --output_csv {workspace}/eval_results/fvd_table.csv
""")


def main():
    parser = argparse.ArgumentParser(description="평가 워크스페이스 초기화")
    parser.add_argument(
        "--workspace", type=Path,
        default=Path("/home/dongwoo43/qfm/eval_workspace"),
        help="워크스페이스 루트 경로"
    )
    parser.add_argument(
        "--qsfm_output", type=Path,
        default=Path("outputs/qsfm_lora"),
        help="QSFM 훈련 output 디렉토리"
    )
    parser.add_argument(
        "--ground_truth", type=Path,
        default=Path("/home/dongwoo43/qfm/qsfm_data/scenes"),
        help="Ground truth 비디오 소스 디렉토리"
    )
    args = parser.parse_args()

    create_workspace(args.workspace, args.qsfm_output, args.ground_truth)


if __name__ == "__main__":
    main()
