#!/usr/bin/env python
"""
마스터 평가 스크립트  (논문 Table 1)
======================================
워크스페이스의 모든 모델 폴더에 대해:
  - CLIPSIM     (텍스트-비디오 CLIP 유사도)
  - Temporal Consistency (샷간 CLIP 코사인 유사도)
  - FVD         (선택, compute_fvd.py 기반)

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer

    python scripts/run_master_eval.py \\
        --workspace /home/dongwoo43/qfm/eval_workspace \\
        --output_csv /home/dongwoo43/qfm/eval_workspace/eval_results/table1.csv

    # FVD 포함
    python scripts/run_master_eval.py \\
        --workspace /home/dongwoo43/qfm/eval_workspace \\
        --compute_fvd \\
        --output_csv /home/dongwoo43/qfm/eval_workspace/eval_results/table1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 표준 평가 프롬프트 (prompts.json에서 로드, fallback)
DEFAULT_PROMPTS = [
    "A cartoon rabbit waddles through an open meadow as small animated birds circle overhead.",
    "A large fluffy rabbit sits near a pond surrounded by trees in an animated nature scene.",
    "Three squirrels fly through the air over a forest as a giant rabbit watches with curiosity.",
    "An animated rabbit chases a small rodent through tall grass in a colorful cartoon forest.",
]


# ── 유틸리티 ─────────────────────────────────────────────────────────────
def _to_tensor(feat) -> torch.Tensor:
    """CLIP 출력 → Tensor (버전 호환)."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "last_hidden_state"):
        return feat.last_hidden_state[:, 0]
    raise ValueError(f"알 수 없는 CLIP 출력: {type(feat)}")


def load_video_frames(path: Path, n_frames: int = 8) -> np.ndarray:
    """비디오 → (N, H, W, 3) uint8."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"프레임 없음: {path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)


def get_video_clip_feat(model, processor, path: Path, n_frames: int = 8) -> torch.Tensor:
    """비디오 → CLIP 임베딩 (1, D)."""
    from PIL import Image

    frames = load_video_frames(path, n_frames)
    pil = [Image.fromarray(f) for f in frames]
    with torch.no_grad():
        inp = processor(images=pil, return_tensors="pt", padding=True).to(DEVICE)
        feats = F.normalize(_to_tensor(model.get_image_features(**inp)), dim=-1)
        return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)


# ── 모델별 지표 계산 ─────────────────────────────────────────────────────
def eval_model_dir(
    model_dir: Path,
    prompts: list[str],
    clip_model,
    clip_proc,
    n_frames: int = 8,
) -> dict:
    """
    단일 모델 폴더 평가:
      - CLIPSIM (프롬프트 ↔ 비디오)
      - Temporal Consistency (1번 샷 ↔ 각 샷)
    """
    # 비디오 파일 탐색 (combined 제외)
    video_paths = sorted([
        p for p in model_dir.glob("*.mp4")
        if "combined" not in p.name.lower()
    ])

    if not video_paths:
        return {
            "clipsim": float("nan"),
            "consistency": float("nan"),
            "consistency_drop": float("nan"),
            "n_shots": 0,
        }

    n = min(len(prompts), len(video_paths))
    video_paths = video_paths[:n]
    prompts_used = prompts[:n]

    # ── CLIPSIM ────────────────────────────────────────────────────
    with torch.no_grad():
        text_inp = clip_proc(
            text=prompts_used, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        text_feats = F.normalize(_to_tensor(clip_model.get_text_features(**text_inp)), dim=-1)

    clipsims = []
    video_feats = []
    for i, vp in enumerate(video_paths):
        try:
            vfeat = get_video_clip_feat(clip_model, clip_proc, vp, n_frames)
            sim = (text_feats[[i]] * vfeat).sum(dim=-1).item()
            clipsims.append(sim)
            video_feats.append(vfeat)
        except Exception as e:
            print(f"    ⚠ {vp.name}: {e}")

    # ── Temporal Consistency ────────────────────────────────────────
    sims_tc = []
    base = video_feats[0] if video_feats else None
    for vf in video_feats:
        if base is not None:
            sims_tc.append((base * vf).sum(dim=-1).item())

    return {
        "clipsim": float(np.mean(clipsims)) if clipsims else float("nan"),
        "consistency": float(np.mean(sims_tc)) if sims_tc else float("nan"),
        "consistency_drop": float(sims_tc[0] - sims_tc[-1]) if len(sims_tc) > 1 else 0.0,
        "n_shots": n,
    }


# ── FVD 연동 ─────────────────────────────────────────────────────────────
def compute_fvd_for_model(
    real_dir: Path, model_dir: Path, extractor
) -> float:
    """compute_fvd.py 의 함수를 직접 import하여 FVD 계산."""
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_fvd import compute_fvd

    try:
        result = compute_fvd(real_dir, model_dir, extractor)
        return result["fvd"]
    except Exception as e:
        print(f"    ⚠ FVD 실패: {e}")
        return float("nan")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="마스터 평가: Table 1 생성")
    parser.add_argument(
        "--workspace", type=Path,
        default=Path("/home/dongwoo43/qfm/eval_workspace"),
        help="eval_workspace 루트"
    )
    parser.add_argument(
        "--output_csv", type=Path,
        default=None,
        help="결과 CSV 저장 경로 (기본: workspace/eval_results/table1.csv)"
    )
    parser.add_argument("--n_frames",   type=int, default=8)
    parser.add_argument("--compute_fvd", action="store_true",
                        help="FVD 계산 포함 (compute_fvd.py 필요)")
    parser.add_argument("--use_r3d",    action="store_true",
                        help="FVD에 R3D-18 강제 사용")
    args = parser.parse_args()

    # 경로 설정
    workspace = args.workspace
    output_csv = args.output_csv or (workspace / "eval_results" / "table1.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 프롬프트 로드
    prompts_json = workspace / "prompts.json"
    if prompts_json.exists():
        data = json.loads(prompts_json.read_text(encoding="utf-8"))
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    else:
        prompts = DEFAULT_PROMPTS
    print(f"프롬프트 수: {len(prompts)}")

    # 평가 대상 폴더 수집
    model_dirs: dict[str, Path] = {}
    for sub in ["baselines", "qsfm_outputs"]:
        d = workspace / sub
        if d.exists():
            for c in sorted(d.iterdir()):
                if c.is_dir() and any(c.glob("*.mp4")):
                    model_dirs[c.name] = c

    if not model_dirs:
        print(f"⚠ 평가 가능한 비디오 폴더 없음: {workspace}")
        print("  setup_eval_workspace.py 를 먼저 실행하거나 수동으로 비디오를 배치하세요.")
        sys.exit(1)

    print(f"\n평가 대상 모델: {list(model_dirs.keys())}")

    # ── CLIP 로드 ─────────────────────────────────────────────────
    print("\nCLIP 로딩...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # ── FVD 추출기 로드 (선택) ────────────────────────────────────
    fvd_extractor = None
    gt_dir = workspace / "ground_truth"
    if args.compute_fvd:
        print("FVD 추출기 로딩...")
        sys.path.insert(0, str(Path(__file__).parent))
        from compute_fvd import I3DFeatureExtractor
        fvd_extractor = I3DFeatureExtractor(use_fallback=args.use_r3d)

    # ── 평가 루프 ────────────────────────────────────────────────
    all_results = {}
    print("\n" + "="*65)
    for name, mdir in model_dirs.items():
        print(f"\n모델: {name}")
        print(f"  경로: {mdir}")

        metrics = eval_model_dir(mdir, prompts, clip_model, clip_proc, args.n_frames)
        print(f"  CLIPSIM         : {metrics['clipsim']:.4f}")
        print(f"  Consistency     : {metrics['consistency']:.4f}")
        print(f"  Consist. Drop   : {metrics['consistency_drop']:.4f}")
        print(f"  Shot 수         : {metrics['n_shots']}")

        if fvd_extractor is not None and gt_dir.exists():
            fvd = compute_fvd_for_model(gt_dir, mdir, fvd_extractor)
            metrics["fvd"] = fvd
            print(f"  FVD             : {fvd:.2f}")
        else:
            metrics["fvd"] = float("nan")

        all_results[name] = metrics

    # ── Table 1 출력 ────────────────────────────────────────────
    print("\n" + "="*65)
    print("논문 Table 1 (Quality & Temporal Consistency)")
    print("="*65)

    cols = ["CLIPSIM(↑)", "Consistency(↑)", "Drop(↓)", "FVD(↓)", "K"]
    header = f"  {'Method':28s}  " + "  ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # QSFM Full을 기준으로 정렬 (있으면)
    order = sorted(
        all_results.keys(),
        key=lambda x: (
            0 if "qsfm_full" in x else (1 if "qsfm" in x else 2),
            all_results[x].get("clipsim", -1) * -1
        )
    )

    for name in order:
        m = all_results[name]
        tag = "(Ours)" if "qsfm_full" in name else ("(Ablation)" if "qsfm" in name else "")
        label = f"{name} {tag}"[:28]
        row = (
            f"  {label:28s}  "
            f"{m.get('clipsim', float('nan')):>14.4f}  "
            f"{m.get('consistency', float('nan')):>14.4f}  "
            f"{m.get('consistency_drop', float('nan')):>8.4f}  "
            f"{m.get('fvd', float('nan')):>8.2f}  "
            f"{m.get('n_shots', 0):>3d}"
        )
        print(row)

    # ── CSV 저장 ─────────────────────────────────────────────────
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("model,clipsim,consistency,consistency_drop,fvd,n_shots\n")
        for name in order:
            m = all_results[name]
            f.write(
                f"{name},"
                f"{m.get('clipsim', float('nan')):.4f},"
                f"{m.get('consistency', float('nan')):.4f},"
                f"{m.get('consistency_drop', float('nan')):.4f},"
                f"{m.get('fvd', float('nan')):.4f},"
                f"{m.get('n_shots', 0)}\n"
            )

    # JSON도 저장
    json_out = output_csv.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ CSV 저장: {output_csv}")
    print(f"   JSON 저장: {json_out}")


if __name__ == "__main__":
    main()
