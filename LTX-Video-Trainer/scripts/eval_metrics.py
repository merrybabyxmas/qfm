#!/usr/bin/env python
"""
QSFM Evaluation Metrics
========================
생성된 비디오에 대해 다음 지표를 계산합니다:

  1. CLIPSIM         : 텍스트 프롬프트 ↔ 비디오 프레임 CLIP 코사인 유사도
  2. Shot-wise CLIP  : 첫 번째 샷 ↔ 각 샷의 CLIP 임베딩 유사도 (시간적 일관성)
  3. FVD (선택)      : pytorch-fid 설치 시 Fréchet Video Distance 계산

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer

    # 기본: 가장 최근 validation 폴더 자동 탐색
    python scripts/eval_metrics.py \\
        --output_dir outputs/qsfm_lora \\
        --prompts "A cartoon rabbit waddles..." "A large fluffy rabbit..." \\
        "Three squirrels fly..." "An animated rabbit chases..."

    # 특정 step 지정
    python scripts/eval_metrics.py \\
        --videos_dir outputs/qsfm_lora/validation/step_001000 \\
        --prompts "A cartoon rabbit waddles..." ...
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


# ── 의존성 로드 ──────────────────────────────────────────────────────────
def _load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor
    except ImportError:
        raise RuntimeError(
            "transformers 미설치. 실행: pip install transformers"
        )


def _to_tensor(feat) -> torch.Tensor:
    """CLIPModel get_*_features 반환값 → Tensor 정규화 (버전 호환)."""
    if isinstance(feat, torch.Tensor):
        return feat
    # BaseModelOutputWithPooling 등 dataclass 처리
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "last_hidden_state"):
        return feat.last_hidden_state[:, 0]
    raise ValueError(f"알 수 없는 CLIP 출력 타입: {type(feat)}")


def _load_video_frames(video_path: Path, n_frames: int = 8) -> np.ndarray:
    """비디오에서 N 프레임 균등 샘플링. 반환: (N, H, W, 3) uint8."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 미설치. 실행: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"프레임 없음: {video_path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"프레임 로드 실패: {video_path}")
    return np.stack(frames)   # (N, H, W, 3)


def _get_video_clip_feat(model, processor, video_path: Path, n_frames: int = 8) -> torch.Tensor:
    """비디오 → CLIP 이미지 임베딩 (평균 풀링). 반환: (1, D)."""
    from PIL import Image
    frames = _load_video_frames(video_path, n_frames)
    pil_frames = [Image.fromarray(f) for f in frames]

    with torch.no_grad():
        inputs = processor(images=pil_frames, return_tensors="pt", padding=True).to(DEVICE)
        feats = _to_tensor(model.get_image_features(**inputs))   # (N, D)
        feats = F.normalize(feats, dim=-1)
        video_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
    return video_feat


# ── 지표 1: CLIPSIM ──────────────────────────────────────────────────────
def compute_clipsim(
    model, processor,
    prompts: list[str],
    video_paths: list[Path],
    n_frames: int = 8,
) -> dict[str, float]:
    """
    CLIPSIM: 텍스트 프롬프트 ↔ 비디오 프레임 CLIP 코사인 유사도.
    논문 Table 1에서 WebVid 실험용 주요 지표.
    """
    print("\n── CLIPSIM (Text-Video Similarity) ───────────────────────────────────")
    scores: dict[str, float] = {}

    with torch.no_grad():
        text_inputs = processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        text_feats = _to_tensor(model.get_text_features(**text_inputs))   # (P, D)
        text_feats = F.normalize(text_feats, dim=-1)

    for i, (prompt, vp) in enumerate(zip(prompts, video_paths)):
        try:
            video_feat = _get_video_clip_feat(model, processor, vp, n_frames)  # (1, D)
            sim = (text_feats[[i]] * video_feat).sum(dim=-1).item()
            scores[vp.name] = sim
            print(f"  [{vp.stem:40s}]  CLIPSIM = {sim:.4f}")
        except Exception as e:
            print(f"  [{vp.name}] 오류: {e}")
            scores[vp.name] = float("nan")

    valid = [v for v in scores.values() if not np.isnan(v)]
    scores["__mean__"] = float(np.mean(valid)) if valid else float("nan")
    print(f"  {'Mean CLIPSIM':44s}= {scores['__mean__']:.4f}")
    return scores


# ── 지표 2: Shot-wise Temporal Consistency ───────────────────────────────
def compute_temporal_consistency(
    model, processor,
    video_paths: list[Path],
    n_frames: int = 8,
) -> dict[str, float]:
    """
    Shot-wise CLIP Cosine Similarity:
      각 샷(Shot)의 비디오 CLIP 임베딩을 1번 샷과 비교.
      QSFM 얽힘(Entanglement) 구조 덕분에 값이 고전 모델보다 덜 떨어져야 함.

    논문 Table 1 & Figure 1에서 QSFM의 핵심 어드밴티지 지표.
    """
    print("\n── Shot-wise Temporal Consistency (CLIP Cosine Sim) ───────────────────")

    shot_feats: list[torch.Tensor | None] = []
    for vp in video_paths:
        try:
            feat = _get_video_clip_feat(model, processor, vp, n_frames)
            shot_feats.append(feat)
        except Exception as e:
            print(f"  [{vp.name}] 오류: {e}")
            shot_feats.append(None)

    # 첫 번째 유효 샷을 기준으로 비교
    base_feat = next((f for f in shot_feats if f is not None), None)
    if base_feat is None:
        return {"error": "유효한 샷 없음"}

    results: dict[str, float] = {}
    sims: list[float] = []
    for i, feat in enumerate(shot_feats):
        if feat is not None:
            sim = (base_feat * feat).sum(dim=-1).item()
            key = f"shot_{i+1:02d}_vs_shot_01"
            results[key] = sim
            sims.append(sim)
            print(f"  Shot {i+1:2d} vs Shot 01 :  {sim:.4f}")
        else:
            results[f"shot_{i+1:02d}_vs_shot_01"] = float("nan")

    results["__mean_consistency__"] = float(np.mean(sims)) if sims else float("nan")
    results["__consistency_drop__"] = float(sims[0] - sims[-1]) if len(sims) > 1 else 0.0
    print(f"  {'Mean Consistency':42s}= {results['__mean_consistency__']:.4f}")
    print(f"  {'Consistency Drop (shot1 → last)':42s}= {results['__consistency_drop__']:.4f}")
    return results


# ── 지표 3: FVD (선택적) ─────────────────────────────────────────────────
def compute_fvd_if_available(real_dir: Path | None, fake_dir: Path) -> float | None:
    """
    FVD 계산 (pytorch-fid 또는 stylegan-v 기반 I3D).
    패키지가 없으면 None 반환.
    현재는 기반 패키지 감지만 수행, 실제 FVD는 별도 스크립트 권장.
    """
    try:
        import torchvision  # noqa: F401
        print("\n── FVD ────────────────────────────────────────────────────────────────")
        print("  ⚠️  FVD 계산은 I3D pretrained 모델 필요.")
        print("     stylegan-v 또는 pytorch-fid + i3d 설치 후 별도 실행하세요.")
        print("     참고: https://github.com/universome/stylegan-v  (FVD 공식 구현)")
        return None
    except ImportError:
        return None


# ── 결과 저장 ────────────────────────────────────────────────────────────
def save_results(
    output_path: Path,
    clipsim: dict[str, float],
    consistency: dict[str, float],
    step: str = "",
) -> None:
    data = {
        "step": step,
        "clipsim": clipsim,
        "temporal_consistency": consistency,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 텍스트 요약도 저장
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("QSFM Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        if step:
            f.write(f"Step: {step}\n\n")
        f.write("CLIPSIM (Text-Video Similarity):\n")
        for k, v in clipsim.items():
            if not k.startswith("__"):
                f.write(f"  {k}: {v:.4f}\n")
        f.write(f"  Mean: {clipsim.get('__mean__', float('nan')):.4f}\n\n")

        f.write("Temporal Consistency (Shot-wise CLIP Sim):\n")
        for k, v in consistency.items():
            if not k.startswith("__"):
                f.write(f"  {k}: {v:.4f}\n")
        f.write(f"  Mean Consistency:  {consistency.get('__mean_consistency__', float('nan')):.4f}\n")
        f.write(f"  Consistency Drop:  {consistency.get('__consistency_drop__', float('nan')):.4f}\n")

    print(f"\n✅ 결과 저장: {output_path}")
    print(f"   텍스트 요약: {txt_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QSFM Evaluation Metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--videos_dir", type=Path,
        help="개별 샷 비디오(.mp4)가 있는 디렉토리 (combined 제외)"
    )
    group.add_argument(
        "--output_dir", type=Path,
        help="훈련 output 디렉토리. 가장 최근 validation step 자동 탐색."
    )
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        help="샷별 텍스트 프롬프트 (비디오 순서와 동일)"
    )
    parser.add_argument(
        "--n_frames", type=int, default=8,
        help="비디오당 샘플링할 프레임 수 (기본 8)"
    )
    parser.add_argument(
        "--step", type=str, default="",
        help="결과에 기록할 학습 step 번호"
    )
    parser.add_argument(
        "--real_dir", type=Path, default=None,
        help="FVD 계산용 실제 비디오 디렉토리 (선택)"
    )
    args = parser.parse_args()

    # ── videos_dir 결정 ───────────────────────────────────────────────
    if args.videos_dir:
        videos_dir = args.videos_dir
    else:
        # output_dir에서 가장 최근 validation step 탐색
        val_dirs = sorted(args.output_dir.glob("validation/step_*"))
        if not val_dirs:
            print(f"오류: {args.output_dir}/validation/step_* 를 찾을 수 없습니다.")
            sys.exit(1)
        videos_dir = val_dirs[-1]
        print(f"자동 탐색된 validation 디렉토리: {videos_dir}")

    # combined 비디오 제외하고 개별 샷만
    video_paths = sorted([
        p for p in videos_dir.glob("*.mp4")
        if "combined" not in p.name.lower()
    ])

    if not video_paths:
        print(f"오류: {videos_dir} 에 비디오 없음 (combined 제외)")
        sys.exit(1)

    n = min(len(args.prompts), len(video_paths))
    prompts = args.prompts[:n]
    video_paths = video_paths[:n]

    print("=" * 70)
    print("QSFM Evaluation Metrics")
    print(f"  Videos dir : {videos_dir}")
    print(f"  Shot count : {n}")
    print(f"  Device     : {DEVICE}")
    print("=" * 70)

    # ── CLIP 로드 ─────────────────────────────────────────────────────
    print("\nCLIP 모델 로딩 중 (openai/clip-vit-base-patch32)...")
    try:
        clip_model, clip_proc = _load_clip()
        print(f"  완료: {DEVICE}")
    except RuntimeError as e:
        print(f"  실패: {e}")
        sys.exit(1)

    # ── 지표 계산 ─────────────────────────────────────────────────────
    clipsim = compute_clipsim(clip_model, clip_proc, prompts, video_paths, args.n_frames)
    consistency = compute_temporal_consistency(clip_model, clip_proc, video_paths, args.n_frames)
    compute_fvd_if_available(args.real_dir, videos_dir)

    # ── 최종 요약 출력 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("최종 요약")
    print(f"  CLIPSIM (mean)         : {clipsim.get('__mean__', float('nan')):.4f}")
    print(f"  Temporal Consistency   : {consistency.get('__mean_consistency__', float('nan')):.4f}")
    print(f"  Consistency Drop       : {consistency.get('__consistency_drop__', float('nan')):.4f}")
    print("=" * 70)

    # ── 결과 저장 ─────────────────────────────────────────────────────
    out_json = videos_dir / "eval_metrics.json"
    save_results(out_json, clipsim, consistency, step=args.step or videos_dir.name)


if __name__ == "__main__":
    main()
