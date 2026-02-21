"""
QSFM Training Strategy
======================
LTX-Video-Trainer 의 TrainingStrategy 인터페이스를 구현하여
Quantum Superposition Flow Matching 학습을 수행.

- 배치 내 K개 샘플을 K개 "샷(Shot)"으로 취급
- QSFM 모듈이 K개 잠재 벡터를 단일 양자 밀도 행렬로 인코딩
- Phase B 포워드 프로세스로 노이즈 주입
- Phase C/D 역방향 PQC 채널로 예측
- Phase E Hilbert-Schmidt 손실 + 기존 Flow Matching 손실 조합
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors
from ltxv_trainer.qsfm import QSFMModule
from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.training_strategies import TrainingBatch, TrainingStrategy

DEFAULT_FPS = 24


class QSFMTrainingStrategy(TrainingStrategy):
    """
    QSFM 학습 전략.

    배치 크기 = K (샷 수). 배치 내 모든 샘플이 하나의 Multi-shot 시퀀스를 구성.

    손실 함수:
        L = α · L_QSFM + (1 − α) · L_flow_matching
    """

    def __init__(
        self,
        conditioning_config: ConditioningConfig,
        qsfm_module: QSFMModule,
        qsfm_loss_weight: float = 0.5,
    ) -> None:
        super().__init__(conditioning_config)
        self.qsfm = qsfm_module
        self.qsfm_loss_weight = qsfm_loss_weight
        logger.info(
            f"[QSFM] Module initialized: D={qsfm_module.D}, "
            f"n_idx={qsfm_module.n_idx_qubits}, n_latent={qsfm_module.n_latent_qubits}, "
            f"loss_weight={qsfm_loss_weight}"
        )

    # ------------------------------------------------------------------
    def get_data_sources(self) -> list[str]:
        return ["latents", "conditions"]

    # ------------------------------------------------------------------
    def prepare_batch(
        self, batch: dict[str, Any], timestep_sampler: TimestepSampler
    ) -> TrainingBatch:
        """
        배치 준비 : K개 잠재 벡터 → QSFM 전처리 + 기존 Flow Matching 노이즈.
        """
        latents_dict = batch["latents"]
        target_latents = latents_dict["latents"]         # (K, seq_len, channels)

        K = target_latents.shape[0]
        latent_frames = latents_dict["num_frames"][0].item()
        latent_height = latents_dict["height"][0].item()
        latent_width  = latents_dict["width"][0].item()

        fps_raw = latents_dict.get("fps", None)
        fps = fps_raw[0].item() if fps_raw is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        prompt_embeds        = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        device = target_latents.device

        # QSFM 모듈을 latent와 동일한 device로 이동 (항상 이동하여 mismatch 방지)
        qsfm_device = next(iter(self.qsfm.parameters())).device
        if qsfm_device.type != device.type or qsfm_device.index != (device.index or 0):
            self.qsfm = self.qsfm.to(device)

        # ── Phase B : 양자 포워드 프로세스 타임스텝 샘플링 ─────────
        # QSFM 타임스텝 t ∈ [0,1]
        # 기존 Flow Matching 시그마와 동일한 값 공유
        sigmas = timestep_sampler.sample_for(target_latents)  # (K, 1, 1) or (K,)
        sigmas_flat = sigmas.view(-1)  # (K,) → QSFM t = σ
        qsfm_t = sigmas_flat.clamp(0.0, 1.0)

        # ── 기존 Flow Matching 노이즈 주입 ─────────────────────────
        noise = torch.randn_like(target_latents)
        sigmas_bcast = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_bcast) * target_latents + sigmas_bcast * noise
        targets = noise - target_latents

        # ── QSFM 포워드 ─────────────────────────────────────────────
        # K개 샷을 각각 (K, latent_dim) 형태의 잠재 벡터로 압축
        # seq_len 차원을 평균 풀링으로 축소
        latent_dim = target_latents.shape[-1]  # channels
        # (K, seq_len, channels) → (K, channels)  via mean over seq_len
        z_shots = target_latents.mean(dim=1)   # (K, channels)

        # QSFM 손실 계산 (그래디언트 흐름 유지)
        # QSFM 모듈은 float32로 동작; 잠재 벡터도 float32로 변환
        z_shots_fp32 = z_shots.float()
        latents_list = [z_shots_fp32[[k]] for k in range(K)]  # K × (1, channels)
        t_batch = qsfm_t.unsqueeze(0).expand(1, K).mean(dim=-1)  # scalar → (1,)
        t_single = t_batch[:1]  # (1,)

        # VRAM & latency 추적
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        # QSFM는 batch=1로 처리 (K개 샷이 단일 양자 상태)
        _, _, qsfm_loss = self.qsfm.training_forward(latents_list, t_single)
        # qsfm_loss는 Tensor scalar — backward 에서 사용 가능

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        qsfm_latency_ms = (time.perf_counter() - t0) * 1000
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        )
        logger.debug(
            f"[QSFM] qsfm_latency={qsfm_latency_ms:.1f}ms, "
            f"peak_vram={peak_vram_mb:.1f}MB"
        )

        # 조건 마스크 (conditioning 없음)
        conditioning_mask = torch.zeros(
            K, noisy_latents.shape[1], dtype=torch.bool, device=device
        )

        sampled_timestep_values = torch.round(sigmas_flat * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(
            conditioning_mask, sampled_timestep_values
        )

        rope_scale = get_rope_scale_factors(fps)

        tb = TrainingBatch(
            latents=noisy_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas_bcast,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale,
            video_coords=None,
        )
        # QSFM 손실을 배치에 첨부
        tb.__dict__["qsfm_loss"] = qsfm_loss
        return tb

    # ------------------------------------------------------------------
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """
        Combined loss:
            L = (1 − α) · L_flow + α · L_QSFM

        α = qsfm_loss_weight
        """
        # 기존 Flow Matching MSE 손실
        loss_fm = (model_pred - batch.targets).pow(2)
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
        loss_fm = loss_fm.mul(loss_mask).div(loss_mask.mean()).mean()

        # QSFM 손실 (prepare_batch 에서 이미 계산됨)
        qsfm_loss = batch.__dict__.get("qsfm_loss", torch.tensor(0.0, device=model_pred.device))

        total = (1.0 - self.qsfm_loss_weight) * loss_fm + self.qsfm_loss_weight * qsfm_loss

        logger.debug(
            f"[QSFM] loss_fm={loss_fm.item():.4f}, "
            f"qsfm_loss={qsfm_loss.item():.4f}, "
            f"total={total.item():.4f}"
        )
        return total
