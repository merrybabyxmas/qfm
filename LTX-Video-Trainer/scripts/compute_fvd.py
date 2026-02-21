#!/usr/bin/env python
"""
FVD (Fréchet Video Distance) 계산기
=====================================
공식 I3D (Kinetics-400) 또는 R3D-18(fallback)을 사용하여 FVD를 계산.

FVD = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2√(Σ_r·Σ_g))

표준 I3D (stylegan-v 배포판) 자동 다운로드 → 실패 시 torchvision R3D-18 사용.

Usage:
    conda activate afm
    cd /home/dongwoo43/qfm/LTX-Video-Trainer

    # 두 폴더 비교 (실제 vs 생성)
    python scripts/compute_fvd.py \\
        --real_dir /path/to/real_videos \\
        --fake_dir /path/to/generated_videos \\
        --n_frames 16 \\
        --resolution 224

    # 전체 워크스페이스 일괄 평가
    python scripts/compute_fvd.py \\
        --workspace /home/dongwoo43/qfm/eval_workspace \\
        --output_csv eval_workspace/eval_results/fvd_table.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# I3D TorchScript (stylegan-v 공식 배포)
I3D_TORCHSCRIPT_URL = (
    "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
)
I3D_CACHE = Path.home() / ".cache" / "qsfm" / "i3d_torchscript.pt"


# ── 비디오 로더 ─────────────────────────────────────────────────────────
def load_video_tensor(
    path: Path, n_frames: int = 16, resolution: int = 224
) -> torch.Tensor:
    """
    MP4 → (C, T, H, W) float32 [0,1] on CPU.
    C=3 (RGB), T=n_frames (균등 샘플링), H=W=resolution.
    """
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
            frame = cv2.resize(frame, (resolution, resolution))
            frames.append(frame)
    cap.release()

    # 부족한 프레임 마지막으로 패딩
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((resolution, resolution, 3), dtype=np.uint8))

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (3, T, H, W)
    return tensor


def load_video_folder(
    folder: Path,
    n_frames: int = 16,
    resolution: int = 224,
    max_videos: int = 200,
    pattern: str = "*.mp4",
    exclude: str = "combined",
) -> torch.Tensor:
    """
    폴더의 모든 MP4 → (N, C, T, H, W).
    combined 파일 제외, max_videos 개 제한.
    """
    paths = sorted([
        p for p in folder.glob(pattern)
        if exclude not in p.name.lower()
    ])[:max_videos]

    if not paths:
        raise ValueError(f"비디오 없음: {folder}")

    tensors = []
    for p in paths:
        try:
            t = load_video_tensor(p, n_frames, resolution)
            tensors.append(t)
        except Exception as e:
            print(f"  ⚠ 로드 실패 {p.name}: {e}")

    if not tensors:
        raise ValueError(f"유효한 비디오 없음: {folder}")

    return torch.stack(tensors, dim=0)  # (N, 3, T, H, W)


# ── 특징 추출기 ──────────────────────────────────────────────────────────
class I3DFeatureExtractor(nn.Module):
    """
    Kinetics-400 사전학습 I3D (stylegan-v TorchScript) 기반 특징 추출.
    다운로드 실패 시 R3D-18(torchvision) fallback.
    """

    def __init__(self, cache_path: Path = I3D_CACHE, use_fallback: bool = False):
        super().__init__()
        self.use_i3d = False

        if not use_fallback:
            self.use_i3d = self._try_load_i3d(cache_path)

        if not self.use_i3d:
            self._load_r3d()

    def _try_load_i3d(self, cache_path: Path) -> bool:
        """I3D TorchScript 로드 시도. 성공 여부 반환."""
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  I3D 다운로드 시도: {I3D_TORCHSCRIPT_URL}")
            try:
                urllib.request.urlretrieve(I3D_TORCHSCRIPT_URL, str(cache_path))
                print(f"  I3D 저장: {cache_path}")
            except Exception as e:
                print(f"  I3D 다운로드 실패 ({e}) → R3D-18 fallback 사용")
                return False

        try:
            self.i3d = torch.jit.load(str(cache_path), map_location="cpu")
            self.i3d.eval()
            print("  I3D (Kinetics-400, stylegan-v) 로드 ✅")
            return True
        except Exception as e:
            print(f"  I3D 로드 실패 ({e}) → R3D-18 fallback 사용")
            return False

    def _load_r3d(self):
        """R3D-18 (torchvision) fallback. 특징 차원: 512."""
        from torchvision.models.video import R3D_18_Weights, r3d_18

        print("  R3D-18 (torchvision, Kinetics-400) 로드 ✅  [I3D fallback]")
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # 분류 헤드 제거 → avgpool 이후 512-D 특징 사용
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.encoder.eval()
        self.feat_dim = 512

    @torch.no_grad()
    def extract(self, videos: torch.Tensor, batch_size: int = 4) -> np.ndarray:
        """
        Args:
            videos : (N, 3, T, H, W) float32 [0,1]
        Returns:
            feats  : (N, feat_dim) float32 numpy
        """
        videos = videos.to(DEVICE)
        # Normalize (ImageNet mean/std — R3D-18 & I3D 동일)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1, 1)
        videos = (videos - mean) / std

        all_feats = []
        N = videos.shape[0]
        for i in range(0, N, batch_size):
            batch = videos[i:i + batch_size]
            if self.use_i3d:
                feat = self.i3d(batch)   # I3D TorchScript 출력 구조에 따라 다름
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]
                feat = feat.view(feat.shape[0], -1)
            else:
                feat = self.encoder(batch)     # (B, 512, 1, 1, 1) 또는 (B, 512)
                feat = feat.view(feat.shape[0], -1)
            all_feats.append(feat.cpu().float().numpy())

        return np.concatenate(all_feats, axis=0)


# ── FVD 계산 ─────────────────────────────────────────────────────────────
def compute_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """특징 벡터 → (μ, Σ)."""
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    if sigma.ndim == 0:               # 샘플 1개 edge case
        sigma = np.array([[sigma]])
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Fréchet Distance:
      FD = ||μ1 − μ2||² + Tr(Σ1 + Σ2 − 2·sqrt(Σ1·Σ2))
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    # 수치 안정성: 대각 정규화
    covmean, _ = sqrtm(sigma1 @ sigma2 + eps * np.eye(sigma1.shape[0]), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fd))


def compute_fvd(
    real_dir: Path,
    fake_dir: Path,
    extractor: I3DFeatureExtractor,
    n_frames: int = 16,
    resolution: int = 224,
    max_videos: int = 200,
) -> dict[str, float]:
    """
    두 폴더의 FVD 계산.

    Returns:
        {"fvd": float, "n_real": int, "n_fake": int, "feat_dim": int}
    """
    print(f"\n  실제  : {real_dir.name}  ({max_videos}개 제한)")
    real_videos = load_video_folder(real_dir, n_frames, resolution, max_videos)
    print(f"  생성  : {fake_dir.name}  ({max_videos}개 제한)")
    fake_videos = load_video_folder(fake_dir, n_frames, resolution, max_videos)

    print(f"  real={real_videos.shape}, fake={fake_videos.shape}")

    extractor_device = next(extractor.parameters() if not extractor.use_i3d else iter([])).device \
        if not extractor.use_i3d else DEVICE
    extractor.to(DEVICE)

    print("  특징 추출 중 (real)...")
    real_feats = extractor.extract(real_videos)
    print("  특징 추출 중 (fake)...")
    fake_feats = extractor.extract(fake_videos)

    mu_r, sigma_r = compute_stats(real_feats)
    mu_g, sigma_g = compute_stats(fake_feats)

    fvd = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    return {
        "fvd": fvd,
        "n_real": int(real_videos.shape[0]),
        "n_fake": int(fake_videos.shape[0]),
        "feat_dim": int(real_feats.shape[1]),
    }


# ── 워크스페이스 일괄 평가 ────────────────────────────────────────────────
def eval_workspace(
    workspace: Path,
    extractor: I3DFeatureExtractor,
    n_frames: int,
    resolution: int,
    output_csv: Path,
) -> None:
    """
    eval_workspace/
      ground_truth/  ← 실제 비디오
      baselines/
        model_a/
        model_b/
      qsfm_outputs/
        qsfm_full/
        qsfm_no_hamiltonian/

    각 모델 폴더 FVD vs ground_truth 계산 → CSV.
    """
    gt_dir = workspace / "ground_truth"
    if not gt_dir.exists():
        print(f"오류: ground_truth 디렉토리 없음: {gt_dir}")
        return

    model_dirs: list[Path] = []
    for sub in ["baselines", "qsfm_outputs"]:
        d = workspace / sub
        if d.exists():
            model_dirs.extend([c for c in sorted(d.iterdir()) if c.is_dir()])

    if not model_dirs:
        print("평가할 모델 디렉토리 없음 (baselines/ 또는 qsfm_outputs/)")
        return

    results = []
    for model_dir in model_dirs:
        print(f"\n{'='*60}")
        print(f"모델: {model_dir.name}")
        print(f"{'='*60}")
        try:
            info = compute_fvd(gt_dir, model_dir, extractor, n_frames, resolution)
            info["model"] = model_dir.name
            print(f"  FVD = {info['fvd']:.2f}  (n_real={info['n_real']}, n_fake={info['n_fake']})")
            results.append(info)
        except Exception as e:
            print(f"  ⚠ 실패: {e}")
            results.append({"model": model_dir.name, "fvd": float("nan"), "n_real": 0, "n_fake": 0})

    # ── 결과 테이블 출력 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FVD 비교 결과  (↓ 낮을수록 좋음)")
    print(f"{'='*60}")
    print(f"  {'Model':30s}  FVD")
    print(f"  {'-'*40}")
    for r in sorted(results, key=lambda x: x.get("fvd", float("inf"))):
        print(f"  {r['model']:30s}  {r.get('fvd', float('nan')):.2f}")

    # ── CSV 저장 ─────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("model,fvd,n_real,n_fake,feat_dim\n")
        for r in results:
            f.write(
                f"{r.get('model','')},{r.get('fvd', float('nan')):.4f},"
                f"{r.get('n_real',0)},{r.get('n_fake',0)},{r.get('feat_dim',0)}\n"
            )
    print(f"\n✅ FVD CSV 저장: {output_csv}")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FVD Computation (I3D / R3D-18)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real_dir", type=Path, help="실제 비디오 폴더")
    mode.add_argument("--workspace", type=Path, help="eval_workspace 루트 (일괄 평가)")

    parser.add_argument("--fake_dir",   type=Path, help="생성 비디오 폴더 (--real_dir 사용 시)")
    parser.add_argument("--n_frames",   type=int, default=16, help="프레임 수 (기본 16)")
    parser.add_argument("--resolution", type=int, default=224, help="리사이즈 해상도 (기본 224)")
    parser.add_argument("--max_videos", type=int, default=200, help="폴더당 최대 비디오 수")
    parser.add_argument("--use_r3d",    action="store_true", help="R3D-18 강제 사용 (I3D 스킵)")
    parser.add_argument("--output_csv", type=Path,
                        default=Path("outputs/fvd_results.csv"))
    args = parser.parse_args()

    print("=" * 65)
    print("FVD 계산기  (I3D Kinetics-400 / R3D-18 fallback)")
    print(f"  Device    : {DEVICE}")
    print(f"  n_frames  : {args.n_frames}")
    print(f"  resolution: {args.resolution}")
    print("=" * 65)

    print("\n특징 추출기 로딩...")
    extractor = I3DFeatureExtractor(use_fallback=args.use_r3d)

    if args.workspace:
        eval_workspace(
            args.workspace, extractor,
            args.n_frames, args.resolution, args.output_csv
        )
    else:
        if args.fake_dir is None:
            print("오류: --real_dir 사용 시 --fake_dir 필요")
            sys.exit(1)
        result = compute_fvd(
            args.real_dir, args.fake_dir, extractor,
            args.n_frames, args.resolution, args.max_videos
        )
        print(f"\n  FVD = {result['fvd']:.2f}")
        print(f"  n_real={result['n_real']}, n_fake={result['n_fake']}, "
              f"feat_dim={result['feat_dim']}")

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write("real_dir,fake_dir,fvd,n_real,n_fake\n")
            f.write(f"{args.real_dir},{args.fake_dir},{result['fvd']:.4f},"
                    f"{result['n_real']},{result['n_fake']}\n")
        print(f"  CSV 저장: {args.output_csv}")


if __name__ == "__main__":
    main()
