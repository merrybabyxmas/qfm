#!/usr/bin/env python
"""
Multi-shot Cut-edited Video 평가 지표 (논문 재설계 버전)
==========================================================
컷 편집(Cut-edit) 기반 멀티샷 생성에 특화된 3가지 지표:

1. Global Semantic Cohesion (GSC) ↑
   - 각 샷과 텍스트 프롬프트 간 CLIP 유사도 평균
   - "프롬프트의 주제를 모든 샷에서 유지하는가?"
   - 높을수록 좋음

2. Inter-shot Diversity (ISD) ↑
   - 샷 쌍 간 CLIP 임베딩 코사인 유사도의 역수 (1 - avg_sim)
   - "진짜 컷 전환이 일어났는가?"
   - 적당히 높아야 함 (너무 낮 = 컷 없음, 너무 높 = 무관 영상)

3. Subject Consistency (SC) ↑
   - DINOv2 CLS 임베딩으로 각 샷 첫 프레임 간 동일 주체 보존율
   - "컷이 바뀌어도 주인공은 같은 모습인가?"
   - QSFM의 양자 얽힘이 빛나는 지점

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer
    PYTHONPATH=src python scripts/eval_multi_shot_metrics.py \\
        --shots_dir /path/to/shots/ \\
        --prompts "prompt1" "prompt2" "prompt3" "prompt4"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── CLIP 로더 (공유 캐시) ────────────────────────────────────────────────────
_CLIP_CACHE: dict = {}
_DINO_CACHE: dict = {}


def _load_clip(device: torch.device):
    key = str(device)
    if key not in _CLIP_CACHE:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        _CLIP_CACHE[key] = (model, processor)
    return _CLIP_CACHE[key]


def _load_dino(device: torch.device):
    key = str(device)
    if key not in _DINO_CACHE:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model.eval()
        _DINO_CACHE[key] = (model, processor)
    return _DINO_CACHE[key]


def _to_tensor(feat) -> torch.Tensor:
    """CLIPModel 출력 → Tensor (버전 호환)."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "last_hidden_state"):
        return feat.last_hidden_state[:, 0]
    raise ValueError(f"Unknown feature type: {type(feat)}")


def _load_video_frames(video_path: Path, max_frames: int = 16) -> list[Image.Image]:
    """비디오에서 균등 샘플링으로 프레임 추출."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _get_clip_video_embedding(
    frames: list[Image.Image],
    model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """비디오 프레임들 → 평균 CLIP 이미지 임베딩 (1, D)."""
    if not frames:
        return torch.zeros(1, 512, device=device)

    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = _to_tensor(feats)  # (N_frames, D)
        feats = F.normalize(feats, dim=-1)
        mean_feat = feats.mean(dim=0, keepdim=True)  # (1, D)
        mean_feat = F.normalize(mean_feat, dim=-1)

    return mean_feat


def _get_clip_text_embedding(
    text: str,
    model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """텍스트 → CLIP 텍스트 임베딩 (1, D)."""
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
        feat = _to_tensor(feat)  # (1, D)
        feat = F.normalize(feat, dim=-1)
    return feat


def _get_dino_frame_embedding(
    frame: Image.Image,
    model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """이미지 → DINOv2 CLS 임베딩 (1, D)."""
    inputs = processor(images=frame, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token (1, D)
        cls = F.normalize(cls, dim=-1)
    return cls


# ══════════════════════════════════════════════════════════════════════════════
# 지표 1: Global Semantic Cohesion (GSC)
# ══════════════════════════════════════════════════════════════════════════════
def compute_global_semantic_cohesion(
    shot_paths: list[Path],
    prompts: list[str],
    device: Optional[torch.device] = None,
    max_frames: int = 8,
) -> dict:
    """
    각 샷과 대응 프롬프트 간 CLIP 유사도 평균.

    Args:
        shot_paths: 샷 비디오 경로 리스트 (K개)
        prompts: 대응 텍스트 프롬프트 리스트 (K개 또는 1개 공통)

    Returns:
        dict: {
            'gsc_mean': float,        # 평균 GSC (↑ 좋음)
            'gsc_per_shot': list,     # 샷별 GSC
            'gsc_drop': float,        # 마지막 - 첫 번째 (드리프트)
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_clip(device)

    # 프롬프트가 1개면 모든 샷에 동일 적용
    if len(prompts) == 1:
        prompts = prompts * len(shot_paths)
    elif len(prompts) < len(shot_paths):
        prompts = prompts + [prompts[-1]] * (len(shot_paths) - len(prompts))

    per_shot = []
    for shot_path, prompt in zip(shot_paths, prompts):
        frames = _load_video_frames(shot_path, max_frames)
        if not frames:
            per_shot.append(0.0)
            continue

        vid_emb = _get_clip_video_embedding(frames, model, processor, device)
        txt_emb = _get_clip_text_embedding(prompt, model, processor, device)
        sim = F.cosine_similarity(vid_emb, txt_emb, dim=-1).item()
        per_shot.append(sim)

    gsc_mean = float(np.mean(per_shot)) if per_shot else 0.0
    gsc_drop = float(per_shot[-1] - per_shot[0]) if len(per_shot) > 1 else 0.0

    return {
        "gsc_mean": gsc_mean,
        "gsc_per_shot": per_shot,
        "gsc_drop": gsc_drop,  # 음수 = 마지막 샷으로 갈수록 주제 이탈
    }


# ══════════════════════════════════════════════════════════════════════════════
# 지표 2: Inter-shot Diversity (ISD)
# ══════════════════════════════════════════════════════════════════════════════
def compute_inter_shot_diversity(
    shot_paths: list[Path],
    device: Optional[torch.device] = None,
    max_frames: int = 8,
) -> dict:
    """
    샷 쌍 간 CLIP 임베딩 코사인 유사도의 역수.
    ISD = 1 - avg(sim(shot_i, shot_j) for all i≠j)

    Args:
        shot_paths: 샷 비디오 경로 리스트

    Returns:
        dict: {
            'isd': float,               # Inter-shot Diversity (↑ = 다양)
            'avg_inter_sim': float,     # 평균 샷 간 유사도 (낮을수록 다양)
            'sim_matrix': list,         # K×K 유사도 행렬
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_clip(device)

    # 각 샷 임베딩 계산
    embeddings = []
    for path in shot_paths:
        frames = _load_video_frames(path, max_frames)
        emb = _get_clip_video_embedding(frames, model, processor, device)  # (1, D)
        embeddings.append(emb)

    if len(embeddings) < 2:
        return {"isd": 0.0, "avg_inter_sim": 1.0, "sim_matrix": [[1.0]]}

    # K×K 유사도 행렬
    K = len(embeddings)
    emb_stack = torch.cat(embeddings, dim=0)  # (K, D)
    sim_matrix = torch.mm(emb_stack, emb_stack.T)  # (K, K)
    sim_np = sim_matrix.cpu().numpy().tolist()

    # 대각선 제외한 평균
    off_diag = []
    for i in range(K):
        for j in range(K):
            if i != j:
                off_diag.append(sim_matrix[i, j].item())

    avg_inter_sim = float(np.mean(off_diag)) if off_diag else 1.0
    isd = 1.0 - avg_inter_sim

    return {
        "isd": isd,
        "avg_inter_sim": avg_inter_sim,
        "sim_matrix": sim_np,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 지표 3: Subject Consistency (SC) via DINOv2
# ══════════════════════════════════════════════════════════════════════════════
def compute_subject_consistency(
    shot_paths: list[Path],
    device: Optional[torch.device] = None,
    reference_frame: str = "first",  # "first" or "middle"
) -> dict:
    """
    DINOv2 CLS 임베딩으로 각 샷 대표 프레임 간 주체 일관성 측정.
    SC = 평균(sim(shot_1_frame, shot_i_frame) for i > 1)

    Args:
        shot_paths: 샷 비디오 경로 리스트
        reference_frame: 기준 프레임 ("first" or "middle")

    Returns:
        dict: {
            'sc_mean': float,        # 주체 일관성 (↑ 좋음)
            'sc_per_shot': list,     # 샷별 SC (vs. shot 1)
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dino_model, dino_proc = _load_dino(device)
    except Exception as e:
        print(f"  [SC] DINOv2 로드 실패: {e} → CLIP fallback")
        return _compute_subject_consistency_clip(shot_paths, device)

    def _get_ref_frame(path: Path) -> Optional[Image.Image]:
        frames = _load_video_frames(path, max_frames=16)
        if not frames:
            return None
        idx = 0 if reference_frame == "first" else len(frames) // 2
        return frames[idx]

    # 각 샷 대표 프레임 임베딩
    embeddings = []
    for path in shot_paths:
        frame = _get_ref_frame(path)
        if frame is None:
            embeddings.append(None)
            continue
        emb = _get_dino_frame_embedding(frame, dino_model, dino_proc, device)
        embeddings.append(emb)

    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 2:
        return {"sc_mean": 1.0, "sc_per_shot": [1.0] * len(shot_paths)}

    # Shot 1 기준으로 나머지와 유사도
    ref_emb = valid[0][1]  # shot 1 embedding
    per_shot = []
    for i, emb in enumerate(embeddings):
        if emb is None:
            per_shot.append(0.0)
        elif i == 0:
            per_shot.append(1.0)  # 자기 자신
        else:
            sim = F.cosine_similarity(ref_emb, emb, dim=-1).item()
            per_shot.append(sim)

    sc_mean = float(np.mean(per_shot[1:])) if len(per_shot) > 1 else 1.0

    return {
        "sc_mean": sc_mean,
        "sc_per_shot": per_shot,
    }


def _compute_subject_consistency_clip(
    shot_paths: list[Path],
    device: torch.device,
) -> dict:
    """DINOv2 대신 CLIP 이미지 임베딩으로 SC 계산 (fallback)."""
    model, processor = _load_clip(device)

    def _get_first_frame(path: Path) -> Optional[Image.Image]:
        frames = _load_video_frames(path, max_frames=1)
        return frames[0] if frames else None

    embeddings = []
    for path in shot_paths:
        frame = _get_first_frame(path)
        if frame is None:
            embeddings.append(None)
            continue
        inputs = processor(images=[frame], return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = _to_tensor(feat)
            feat = F.normalize(feat, dim=-1)
        embeddings.append(feat)

    per_shot = []
    ref = next((e for e in embeddings if e is not None), None)
    for i, emb in enumerate(embeddings):
        if emb is None or ref is None:
            per_shot.append(0.0)
        elif i == 0:
            per_shot.append(1.0)
        else:
            per_shot.append(F.cosine_similarity(ref, emb, dim=-1).item())

    sc_mean = float(np.mean(per_shot[1:])) if len(per_shot) > 1 else 1.0
    return {"sc_mean": sc_mean, "sc_per_shot": per_shot}


# ══════════════════════════════════════════════════════════════════════════════
# 통합 평가 함수
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_multi_shot(
    shot_paths: list[Path],
    prompts: list[str],
    device: Optional[torch.device] = None,
    label: str = "model",
) -> dict:
    """
    3가지 지표를 한 번에 계산.

    Returns:
        {
            'label': str,
            'n_shots': int,
            'gsc_mean': float,      # Global Semantic Cohesion ↑
            'gsc_drop': float,      # 마지막-첫 번째 (0에 가까울수록 좋음)
            'isd': float,           # Inter-shot Diversity ↑
            'sc_mean': float,       # Subject Consistency ↑
            'details': {...},
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shot_paths = [Path(p) for p in shot_paths]
    valid_paths = [p for p in shot_paths if p.exists() and p.suffix == ".mp4"]

    if not valid_paths:
        return {"label": label, "n_shots": 0, "gsc_mean": 0.0, "gsc_drop": 0.0,
                "isd": 0.0, "sc_mean": 0.0, "details": {}}

    print(f"\n  [{label}] GSC 계산 중...")
    gsc = compute_global_semantic_cohesion(valid_paths, prompts, device)

    print(f"  [{label}] ISD 계산 중...")
    isd = compute_inter_shot_diversity(valid_paths, device)

    print(f"  [{label}] SC 계산 중 (DINOv2)...")
    sc = compute_subject_consistency(valid_paths, device)

    return {
        "label": label,
        "n_shots": len(valid_paths),
        "gsc_mean": gsc["gsc_mean"],
        "gsc_drop": gsc["gsc_drop"],
        "isd": isd["isd"],
        "sc_mean": sc["sc_mean"],
        "details": {"gsc": gsc, "isd": isd, "sc": sc},
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Multi-shot Cut-edit 평가 지표")
    parser.add_argument("--shots_dir",  type=Path, required=True,
                        help="shot_001.mp4, shot_002.mp4 ... 가 있는 폴더")
    parser.add_argument("--prompts",    nargs="+", default=None,
                        help="각 샷에 대응하는 텍스트 프롬프트 (1개=공통 적용)")
    parser.add_argument("--prompts_json", type=Path, default=None,
                        help="prompts.json 경로 (--prompts 대신 사용)")
    parser.add_argument("--label",      default="model")
    parser.add_argument("--output",     type=Path, default=None)
    args = parser.parse_args()

    # 프롬프트 로드
    if args.prompts_json and args.prompts_json.exists():
        data = json.loads(args.prompts_json.read_text())
        prompts = [p["prompt"] for p in data.get("standard", data.get("all", []))]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = ["a multi-shot video sequence"]

    # 샷 파일 수집
    shot_paths = sorted(args.shots_dir.glob("shot_*.mp4"))
    if not shot_paths:
        print(f"ERROR: {args.shots_dir}에 shot_*.mp4 없음")
        return

    print("=" * 65)
    print(f"Multi-shot 평가: {args.label}")
    print(f"  샷 수   : {len(shot_paths)}")
    print(f"  프롬프트: {len(prompts)}개")
    print("=" * 65)

    result = evaluate_multi_shot(shot_paths, prompts, label=args.label)

    print("\n" + "=" * 65)
    print(f"  GSC (Global Semantic Cohesion) : {result['gsc_mean']:.4f}  ↑")
    print(f"  GSC Drop (마지막-첫번째)        : {result['gsc_drop']:+.4f}  (0에 가까울수록 좋음)")
    print(f"  ISD (Inter-shot Diversity)     : {result['isd']:.4f}  ↑")
    print(f"  SC  (Subject Consistency)      : {result['sc_mean']:.4f}  ↑")
    print("=" * 65)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n저장: {args.output}")


if __name__ == "__main__":
    main()
