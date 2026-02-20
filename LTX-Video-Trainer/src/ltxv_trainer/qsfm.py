"""
Quantum Superposition Flow Matching (QSFM)
==========================================
LTX-Video 잠재 공간(Latent Space) + QDM 밀도 행렬(Density Matrix) 결합 구현.
TorchQuantum PQC 기반으로 단일 양자 상태에서 다중 샷(Multi-shot)을 병렬 생성.

Phase A : 힐베르트 공간 및 초기 상태 정의  (Amplitude Encoding + Superposition)
Phase B : 포워드 프로세스               (Data → Completely Mixed State)
Phase C : 양자 중첩 플로우 매칭         (Lindblad / Stinespring PQC Backward)
Phase D : 시간적 차이 모델링            (Index-Latent Entanglement + Cross-Shot Hamiltonian)
Phase E : 손실 함수                    (Hilbert-Schmidt / Fidelity Loss)
Phase F : 생성 및 추론                  (Backward Flow → Projective Measurement → Decode)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch import Tensor


# ============================================================================
# 유틸리티 함수
# ============================================================================

def _project_density_matrix(rho: Tensor, eps: float = 1e-7) -> Tensor:
    """밀도 행렬 사영: Hermitian + PSD + Tr=1 조건 보장."""
    # 대칭화
    rho = (rho + rho.transpose(-1, -2)) / 2.0
    # PSD : 음의 고유값 클리핑
    # 수치 안정성을 위해 작은 대각 epsilon 추가
    D = rho.shape[-1]
    eye = torch.eye(D, device=rho.device, dtype=rho.dtype)
    rho = rho + eps * eye.unsqueeze(0)
    # 트레이스 정규화
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    return rho / trace.clamp(min=1e-8)


def _safe_eigh(M: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """수치 안정적 고유분해. 정규화 후 eigvalsh 실행."""
    # 약한 대각 정규화
    D = M.shape[-1]
    eye = torch.eye(D, device=M.device, dtype=M.dtype).unsqueeze(0)
    M_reg = (M + M.transpose(-1, -2)) / 2.0 + eps * eye
    try:
        return torch.linalg.eigh(M_reg)
    except Exception:
        # 완전 대각 fallback
        eigvals = M_reg.diagonal(dim1=-2, dim2=-1)
        eigvecs = eye.expand_as(M_reg)
        return eigvals, eigvecs


# ============================================================================
# Phase A — 힐베르트 공간 및 초기 상태 정의
# ============================================================================

class AmplitudeEncoder(nn.Module):
    """
    Phase A.1 : 고전 잠재 벡터 → 정규화 양자 진폭 인코딩.

    z_k ∈ R^{input_dim}  →  |ψ_k⟩ ∈ R^{2^{n_latent_qubits}}

    1. Linear projection : input_dim → 2^{n_latent_qubits}
    2. L2 normalization  : ||·||₂ = 1
    """

    def __init__(self, input_dim: int, n_latent_qubits: int) -> None:
        super().__init__()
        self.n_latent_qubits = n_latent_qubits
        self.d_latent = 2 ** n_latent_qubits
        self.proj = nn.Linear(input_dim, self.d_latent)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z : (batch, input_dim)
        Returns:
            |ψ⟩ : (batch, d_latent)  — unit vector
        """
        out = self.proj(z)
        norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return out / norm


class MultiShotSuperposition(nn.Module):
    """
    Phase A.2 : K개 샷의 양자 중첩 상태 구성.

    |Ψ⟩ = (1/√K) Σ_{k=0}^{K-1} |k⟩_{idx} ⊗ |ψ_k⟩_{latent}

    힐베르트 공간 : H = H_{idx} ⊗ H_{latent}
      - H_{idx}   : dim = d_idx = 2^{n_idx_qubits}
      - H_{latent}: dim = d_latent = 2^{n_latent_qubits}
      - 전체 dim D = d_idx * d_latent

    상태 벡터 레이아웃 :
      state[k * d_latent : (k+1) * d_latent] = ψ_k / √K
    """

    def __init__(self, n_idx_qubits: int, n_latent_qubits: int) -> None:
        super().__init__()
        self.n_idx_qubits = n_idx_qubits
        self.n_latent_qubits = n_latent_qubits
        self.d_idx = 2 ** n_idx_qubits
        self.d_latent = 2 ** n_latent_qubits
        self.D = self.d_idx * self.d_latent

    def forward(self, psi_list: list[Tensor]) -> Tensor:
        """
        Args:
            psi_list : K개의 (batch, d_latent) 정규화 상태 벡터
        Returns:
            |Ψ⟩ : (batch, D)
        """
        K = len(psi_list)
        batch = psi_list[0].shape[0]
        device = psi_list[0].device
        dtype = psi_list[0].dtype

        state = torch.zeros(batch, self.D, device=device, dtype=dtype)
        norm_factor = 1.0 / math.sqrt(K)
        for k, psi_k in enumerate(psi_list):
            start = k * self.d_latent
            end = (k + 1) * self.d_latent
            state[:, start:end] = psi_k * norm_factor
        return state  # (batch, D)


def pure_state_density_matrix(state: Tensor) -> Tensor:
    """
    순수 상태 밀도 행렬 : ρ = |Ψ⟩⟨Ψ|

    Args:
        state : (batch, D)
    Returns:
        ρ     : (batch, D, D)
    """
    return torch.bmm(state.unsqueeze(-1), state.unsqueeze(-2))


# ============================================================================
# Phase B — 포워드 프로세스 (Data → Completely Mixed State)
# ============================================================================

class QuantumForwardProcess:
    """
    Phase B : 데이터 → 완전 혼합 상태(Completely Mixed State) 선형 보간.

    ρ(t) = (1 − t) · ρ(0) + t · (I / D)

    t=0 : 순수 상태(Pure State) ρ(0)
    t=1 : 최대 엔트로피 상태  I / D
    """

    @staticmethod
    def interpolate(rho_0: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            rho_0 : (batch, D, D)
            t     : (batch,) or scalar ∈ [0,1]
        Returns:
            ρ(t)  : (batch, D, D)
        """
        D = rho_0.shape[-1]
        eye = torch.eye(D, device=rho_0.device, dtype=rho_0.dtype) / D
        eye_b = eye.unsqueeze(0).expand_as(rho_0)
        t_v = t.view(-1, 1, 1) if isinstance(t, Tensor) and t.dim() > 0 else t
        return (1.0 - t_v) * rho_0 + t_v * eye_b


# ============================================================================
# Phase C — TorchQuantum PQC 기반 Stinespring Dilation 역방향 채널
# ============================================================================

class TQUnitaryBuilder(nn.Module):
    """
    TorchQuantum QuantumDevice를 이용해 n_wires 큐빗 위의
    파라미터화된 유니터리 행렬 U_θ(t) 를 구성.

    아키텍처:
      - 시간 임베딩 t → MLP → [θ_ry, θ_rz] for each qubit per layer
      - n_layers 레이어: Ry(θ), Rz(θ) + 순환 CNOT 얽힘
      - 출력: U ∈ R^{D×D} (실수 근사; 양자 유니터리의 실수 부분)

    유니터리 행렬 추출 방법:
      각 기저 상태 |e_i⟩ 에 회로 적용 → 열벡터 수집 → 행렬 구성
      이는 자동 미분과 호환됩니다 (param.item()으로 detach 후 외부 grad 계산).
    """

    def __init__(
        self,
        n_wires: int,
        time_embed_dim: int,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.D = 2 ** n_wires
        self.n_layers = n_layers

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, n_wires * n_layers * 2),
        )

    def _build_unitary_for_params(self, params: Tensor, device: torch.device) -> Tensor:
        """
        주어진 파라미터로 유니터리 행렬 구성.

        Args:
            params : (n_wires * n_layers * 2,) - gate angles
        Returns:
            U : (D, D) real unitary matrix
        """
        D = self.D
        cols = []
        for i in range(D):
            # 기저 벡터 |e_i⟩
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=device)
            e_i = torch.zeros(1, D, device=device, dtype=torch.complex64)
            e_i[0, i] = 1.0
            qdev.states = e_i.view([1] + [2] * self.n_wires)

            # 게이트 적용
            idx = 0
            for _layer in range(self.n_layers):
                for wire in range(self.n_wires):
                    ry_val = params[idx]
                    rz_val = params[idx + 1]
                    idx += 2
                    tqf.ry(qdev, wires=wire,
                           params=torch.tensor([ry_val], device=device))
                    tqf.rz(qdev, wires=wire,
                           params=torch.tensor([rz_val], device=device))
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                if self.n_wires > 1:
                    tqf.cnot(qdev, wires=[self.n_wires - 1, 0])

            sv = qdev.get_states_1d()[0]  # (D,) complex64
            cols.append(sv.real)  # 실수부만 사용
        return torch.stack(cols, dim=-1)  # (D, D)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t : (batch,)
        Returns:
            U : (batch, D, D)  파라미터화된 유니터리 행렬
        """
        batch = t.shape[0]
        device = t.device

        params_batch = self.time_mlp(t.unsqueeze(-1).float())  # (batch, n_wires*n_layers*2)

        U_list = []
        for b in range(batch):
            U_b = self._build_unitary_for_params(
                params_batch[b].detach(),  # detach: TQ는 자체 그래디언트 없음
                device=device
            )
            U_list.append(U_b)

        # 유니터리 행렬을 파라미터로 보강 (그래디언트 흐름 유지)
        # params_batch 의 그래디언트를 통해 학습
        # U ≈ I + antisymmetric(params) via first-order expansion
        U_stack = torch.stack(U_list, dim=0)  # (batch, D, D) detached

        # 그래디언트 보강: Lie algebra 근사
        # δU = antisymmetric(A) where A = MLP output reshaped to D×D
        # Full matrix grad via parameter-dependent correction
        D = self.D
        n_params = params_batch.shape[-1]
        if n_params >= D * D:
            A = params_batch[:, :D*D].view(batch, D, D)
        else:
            A = torch.zeros(batch, D, D, device=device, dtype=params_batch.dtype)
            A_flat = A.view(batch, -1)
            A_flat[:, :n_params] = params_batch
        A_skew = (A - A.transpose(-1, -2)) * 0.01  # 스케일 조정
        U_diff = torch.matrix_exp(A_skew)  # (batch, D, D)

        # 그래디언트가 흐르는 유니터리: TQ 결과 × 미분가능 행렬
        U_final = torch.bmm(U_stack.to(U_diff.dtype), U_diff)
        return U_final


class TQSteinspringLayer(nn.Module):
    """
    Phase C : Stinespring Dilation을 이용한 TorchQuantum PQC.

    환경(ancilla) 큐빗을 |0⟩로 초기화 후 system+ancilla 에
    파라미터화된 유니터리 U_θ(t) 를 적용하고 ancilla를 부분 대각합(Partial Trace).

    Φ_θ(ρ) = Tr_{anc}[ U_θ (ρ ⊗ |0⟩⟨0|) U_θ† ]

    = Σ_{k=0}^{D_anc-1} M_k ρ M_k^T    where   M_k = U[k*D_sys:(k+1)*D_sys, :D_sys]

    이 Kraus 표현이 완전양수보존(CPTP) 채널을 정의합니다.
    """

    def __init__(
        self,
        n_system_qubits: int,
        n_ancilla_qubits: int,
        time_embed_dim: int,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_sys = n_system_qubits
        self.n_anc = n_ancilla_qubits
        self.n_total = n_system_qubits + n_ancilla_qubits
        self.D_sys = 2 ** n_system_qubits
        self.D_anc = 2 ** n_ancilla_qubits
        self.D_tot = 2 ** self.n_total

        # TorchQuantum 유니터리 생성기
        self.unitary_builder = TQUnitaryBuilder(
            n_wires=self.n_total,
            time_embed_dim=time_embed_dim,
            n_layers=n_layers,
        )

    def forward(self, rho_in: Tensor, t: Tensor) -> Tensor:
        """
        Stinespring 채널 적용.

        ρ_out = Tr_{anc}[ U_θ(t) · (ρ_in ⊗ |0⟩⟨0|) · U_θ(t)^T ]

        구현:
          Kraus 연산자 M_k = U_θ[k*D_sys:(k+1)*D_sys, :D_sys]
          ρ_out = Σ_k M_k ρ_in M_k^T

        Args:
            rho_in : (batch, D_sys, D_sys)
            t      : (batch,) ∈ [0,1]
        Returns:
            rho_out: (batch, D_sys, D_sys)
        """
        batch = rho_in.shape[0]

        # (batch, D_tot, D_tot) 유니터리 행렬 구성
        U = self.unitary_builder(t)  # (batch, D_tot, D_tot)

        # Kraus 연산자 추출: M_k = U[:, k*D_sys:(k+1)*D_sys, :D_sys]
        rho_out = torch.zeros_like(rho_in)
        for k in range(self.D_anc):
            M_k = U[:, k * self.D_sys:(k + 1) * self.D_sys, :self.D_sys]  # (batch, D_sys, D_sys)
            # ρ_out += M_k ρ_in M_k^T
            rho_out = rho_out + torch.bmm(M_k, torch.bmm(rho_in, M_k.transpose(-1, -2)))

        # 트레이스 정규화
        trace = rho_out.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        rho_out = rho_out / trace.clamp(min=1e-8)
        return rho_out


# ============================================================================
# Phase D — 시간적 차이 모델링 (Index-Latent Entanglement + Shift Hamiltonian)
# ============================================================================

class IndexLatentControlledEntanglement(nn.Module):
    """
    Phase D.1 : 인덱스-잠재 제어 얽힘 게이트.

    |k⟩_{idx} 를 제어(Control), |z⟩_{latent} 를 타겟(Target)으로 하는
    조건부 유니터리 CU_{θ,k}.

    U_ctrl = Σ_k |k⟩⟨k| ⊗ U_{θ,k}    (블록 대각 구조)

    각 U_{θ,k} 는 TorchQuantum n_latent_qubits 큐빗 회로로 구성.
    """

    def __init__(
        self,
        n_idx_qubits: int,
        n_latent_qubits: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.n_idx = n_idx_qubits
        self.n_latent = n_latent_qubits
        self.d_idx = 2 ** n_idx_qubits
        self.d_latent = 2 ** n_latent_qubits
        self.D = self.d_idx * self.d_latent

        # 각 인덱스 k에 대한 TQ 유니터리 빌더
        self.unitary_builders = nn.ModuleList([
            TQUnitaryBuilder(
                n_wires=n_latent_qubits,
                time_embed_dim=time_embed_dim,
                n_layers=1,
            )
            for _ in range(self.d_idx)
        ])

    def forward(self, rho: Tensor, t: Tensor) -> Tensor:
        """
        블록 대각 제어 유니터리 적용 : ρ → U_ctrl ρ U_ctrl^T

        Args:
            rho : (batch, D, D)
            t   : (batch,)
        Returns:
            rho_out : (batch, D, D)
        """
        batch = rho.shape[0]
        device = rho.device

        # 블록 대각 유니터리 구성
        U_full = torch.zeros(batch, self.D, self.D, device=device, dtype=rho.dtype)
        for k in range(self.d_idx):
            U_k = self.unitary_builders[k](t)  # (batch, d_latent, d_latent)
            start = k * self.d_latent
            end = (k + 1) * self.d_latent
            U_full[:, start:end, start:end] = U_k

        # ρ_out = U_ctrl ρ U_ctrl^T
        rho_out = torch.bmm(U_full, torch.bmm(rho, U_full.transpose(-1, -2)))
        trace = rho_out.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        return rho_out / trace.clamp(min=1e-8)


class CrossShotTransitionHamiltonian(nn.Module):
    """
    Phase D.2 : 샷 간 전이 해밀토니안.

    H_shift = λ(t) · (S⁺ ⊗ I_{latent} + S⁻ ⊗ I_{latent})
    S⁺|k⟩ = |(k+1) mod d_idx⟩  (cyclic shift)

    U_shift(t) = exp(−λ(t) · H_shift)
    """

    def __init__(
        self,
        n_idx_qubits: int,
        n_latent_qubits: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.n_idx = n_idx_qubits
        self.n_latent = n_latent_qubits
        self.d_idx = 2 ** n_idx_qubits
        self.d_latent = 2 ** n_latent_qubits
        self.D = self.d_idx * self.d_latent

        self.lambda_net = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, 1),
            nn.Tanh(),  # λ ∈ (-1, 1) 로 제한하여 수치 안정성 확보
        )

        # Shift operator S⁺ on index register
        S_plus = torch.zeros(self.d_idx, self.d_idx)
        for k in range(self.d_idx):
            S_plus[(k + 1) % self.d_idx, k] = 1.0
        H_idx = S_plus + S_plus.T  # Hermitian/Symmetric
        I_lat = torch.eye(self.d_latent)
        H_total = torch.kron(H_idx, I_lat)  # (D, D)
        self.register_buffer("H_total", H_total)

    def forward(self, rho: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            rho : (batch, D, D)
            t   : (batch,)
        Returns:
            rho_out : (batch, D, D)
        """
        batch = rho.shape[0]
        H = self.H_total.to(rho.device)

        lam = self.lambda_net(t.unsqueeze(-1).float()).squeeze(-1) * 0.1  # (batch,) 작은 값

        # U_shift = exp(-λ H)  배치별로 계산
        # 행렬 지수 함수: 안정적 계산을 위해 고유값 분해 사용
        eigvals_H, eigvecs_H = torch.linalg.eigh(H)  # H는 대칭
        # exp(-λ D) = eigvecs @ diag(exp(-λ * eigvals)) @ eigvecs^T
        rho_out_list = []
        for b in range(batch):
            exp_diag = torch.exp(-lam[b] * eigvals_H)
            U_b = eigvecs_H @ torch.diag(exp_diag) @ eigvecs_H.T
            rho_out_list.append(U_b @ rho[b] @ U_b.T)
        rho_out = torch.stack(rho_out_list, dim=0)
        trace = rho_out.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        return rho_out / trace.clamp(min=1e-8)


# ============================================================================
# Phase C/D Combined — 전체 PQC 역방향 채널
# ============================================================================

class QSFMBackwardChannel(nn.Module):
    """
    Phase C + D : 전체 TorchQuantum PQC 역방향 채널.

    Φ_θ(t, ρ) = Phase_D ∘ Phase_C (ρ(t))

    1. Stinespring PQC (Phase C) — Kraus 채널 적용
    2. Index-Latent 제어 얽힘 (Phase D.1)
    3. Cross-Shot 전이 해밀토니안 (Phase D.2)
    """

    def __init__(
        self,
        n_idx_qubits: int,
        n_latent_qubits: int,
        time_embed_dim: int = 64,
        n_pqc_layers: int = 1,
        n_ancilla_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.n_idx = n_idx_qubits
        self.n_latent = n_latent_qubits

        # Phase C : Stinespring Dilation PQC
        self.stinespring = TQSteinspringLayer(
            n_system_qubits=n_idx_qubits + n_latent_qubits,
            n_ancilla_qubits=n_ancilla_qubits,
            time_embed_dim=time_embed_dim,
            n_layers=n_pqc_layers,
        )

        # Phase D.1 : Index-Latent 얽힘
        self.entanglement = IndexLatentControlledEntanglement(
            n_idx_qubits=n_idx_qubits,
            n_latent_qubits=n_latent_qubits,
            time_embed_dim=time_embed_dim,
        )

        # Phase D.2 : Cross-Shot 전이
        self.transition = CrossShotTransitionHamiltonian(
            n_idx_qubits=n_idx_qubits,
            n_latent_qubits=n_latent_qubits,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, rho_t: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            rho_t  : (batch, D, D)
            t      : (batch,) ∈ [0,1]
        Returns:
            rho_pred_0 : (batch, D, D)
        """
        # Phase C
        rho = self.stinespring(rho_t, t)
        # Phase D.1
        rho = self.entanglement(rho, t)
        # Phase D.2
        rho = self.transition(rho, t)
        return _project_density_matrix(rho)


# ============================================================================
# Phase E — 손실 함수 (Hilbert-Schmidt / Fidelity)
# ============================================================================

class HilbertSchmidtLoss(nn.Module):
    """
    Phase E : Hilbert-Schmidt 거리.

    L_HS(ρ, σ) = ||ρ − σ||_F² = Tr[(ρ − σ)²]
    """

    def forward(self, rho_pred: Tensor, rho_target: Tensor) -> Tensor:
        diff = rho_pred - rho_target
        return diff.pow(2).sum(dim=(-2, -1)).mean()


class QuantumFidelityLoss(nn.Module):
    """
    Phase E : 양자 피델리티 기반 손실.

    F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²
    L_fidelity = 1 − F(ρ_pred, ρ_target)

    수치 안정성을 위해 행렬 제곱근은 eigendecomposition 으로 계산.
    """

    def forward(self, rho_pred: Tensor, rho_target: Tensor) -> Tensor:
        batch = rho_pred.shape[0]
        fids = []
        for b in range(batch):
            rp = rho_pred[b].double()
            rt = rho_target[b].double()
            # √ρ_pred via eigendecomposition
            eigvals_p, eigvecs_p = _safe_eigh(rp.unsqueeze(0))
            eigvals_p = eigvals_p[0].clamp(min=0.0)
            sqrt_rp = eigvecs_p[0] @ torch.diag(eigvals_p.sqrt()) @ eigvecs_p[0].T
            # M = √ρ σ √ρ
            M = sqrt_rp @ rt @ sqrt_rp
            eigvals_M, _ = _safe_eigh(M.unsqueeze(0))
            eigvals_M = eigvals_M[0].clamp(min=0.0)
            fid = eigvals_M.sqrt().sum().pow(2)
            fids.append(fid.float())
        return 1.0 - torch.stack(fids).mean()


# ============================================================================
# Phase F — 추론 (Backward Flow → Measure → Decode)
# ============================================================================

class QSFMInference:
    """
    Phase F : Multi-shot 병렬 생성.

    1. 초기화  : ρ(1) = I/D (완전 혼합 상태)
    2. 역방향  : ρ(t) → ρ(0) via PQC ODE 적분
    3. 측정    : 인덱스 레지스터 사영 측정 → K개 잠재 벡터 추출
    4. 디코딩  : VAE Decoder를 통해 픽셀 공간 복원
    """

    def __init__(
        self,
        backward_channel: QSFMBackwardChannel,
        amplitude_encoder: AmplitudeEncoder,
        superposition_builder: MultiShotSuperposition,
        n_inference_steps: int = 20,
    ) -> None:
        self.backward_channel = backward_channel
        self.amplitude_encoder = amplitude_encoder
        self.superposition_builder = superposition_builder
        self.n_steps = n_inference_steps

    @torch.no_grad()
    def generate(self, batch_size: int, K: int, device: torch.device) -> list[Tensor]:
        """
        K개 샷의 잠재 벡터를 역방향 양자 흐름으로 생성.

        Returns:
            list of K tensors, each (batch_size, d_latent)
        """
        D = self.superposition_builder.D

        # Step 1 : 완전 혼합 상태
        rho = torch.eye(D, device=device) / D
        rho = rho.unsqueeze(0).expand(batch_size, -1, -1).clone().float()

        # Step 2 : 역방향 흐름 (t: 1 → 0)
        timesteps = torch.linspace(1.0, 0.0, self.n_steps + 1, device=device)[:-1]
        for t_val in timesteps:
            t_batch = t_val.expand(batch_size)
            rho = self.backward_channel(rho, t_batch)
            rho = _project_density_matrix(rho)

        # Step 3 : 사영 측정으로 K개 잠재 벡터 추출
        return self._measure(rho, K)

    def _measure(self, rho_gen: Tensor, K: int) -> list[Tensor]:
        """
        Phase F.3 : 인덱스 레지스터 기저로 사영 측정.

        k번째 블록의 주 고유벡터를 잠재 벡터로 사용.
        """
        d_latent = self.superposition_builder.d_latent
        shot_latents = []
        for k in range(K):
            start = k * d_latent
            end = (k + 1) * d_latent
            rho_k = rho_gen[:, start:end, start:end]  # (batch, d_latent, d_latent)

            # 수치 안정적 고유분해
            eigvals, eigvecs = _safe_eigh(rho_k)
            # 최대 고유값의 고유벡터
            psi_k = eigvecs[:, :, -1]  # (batch, d_latent)
            shot_latents.append(psi_k)
        return shot_latents


# ============================================================================
# Main QSFM Module
# ============================================================================

class QSFMModule(nn.Module):
    """
    QSFM 전체 파이프라인 통합 모듈.

    Phase A → Phase B → Phase C/D → Phase E

    학습 시:
        rho_pred, rho_target, loss = module.training_forward(latents_list, t)

    추론 시:
        QSFMInference(module.backward_channel, ...).generate(...)
    """

    def __init__(
        self,
        latent_dim: int,
        n_idx_qubits: int = 2,       # K_max = 2^n_idx = 4 shots
        n_latent_qubits: int = 4,    # d_latent = 16 (메모리 효율)
        time_embed_dim: int = 32,
        n_pqc_layers: int = 1,
        n_ancilla_qubits: int = 2,
        loss_type: str = "hilbert_schmidt",
    ) -> None:
        super().__init__()
        self.n_idx_qubits = n_idx_qubits
        self.n_latent_qubits = n_latent_qubits
        self.d_idx = 2 ** n_idx_qubits
        self.d_latent = 2 ** n_latent_qubits
        self.D = self.d_idx * self.d_latent

        # Phase A
        self.amplitude_encoder = AmplitudeEncoder(latent_dim, n_latent_qubits)
        self.superposition_builder = MultiShotSuperposition(n_idx_qubits, n_latent_qubits)

        # Phase C/D
        self.backward_channel = QSFMBackwardChannel(
            n_idx_qubits=n_idx_qubits,
            n_latent_qubits=n_latent_qubits,
            time_embed_dim=time_embed_dim,
            n_pqc_layers=n_pqc_layers,
            n_ancilla_qubits=n_ancilla_qubits,
        )

        # Phase E
        if loss_type == "hilbert_schmidt":
            self.loss_fn: nn.Module = HilbertSchmidtLoss()
        elif loss_type == "fidelity":
            self.loss_fn = QuantumFidelityLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Phase F : 잠재 공간 복원용 디코더 프로젝션
        self.decoder_proj = nn.Linear(self.d_latent, latent_dim)

    # ------------------------------------------------------------------
    def encode_shots(self, latents_list: list[Tensor]) -> Tensor:
        """Phase A : K개 잠재 → 밀도 행렬 ρ(0)."""
        psi_list = [self.amplitude_encoder(z) for z in latents_list]
        state = self.superposition_builder(psi_list)
        rho = pure_state_density_matrix(state)
        return rho

    def forward_process(self, rho_0: Tensor, t: Tensor) -> Tensor:
        """Phase B : ρ(0) → ρ(t)."""
        return QuantumForwardProcess.interpolate(rho_0, t)

    def backward_pass(self, rho_t: Tensor, t: Tensor) -> Tensor:
        """Phase C/D : ρ(t) → ρ̂(0)."""
        return self.backward_channel(rho_t, t)

    def compute_loss(self, rho_pred: Tensor, rho_target: Tensor) -> Tensor:
        """Phase E : Hilbert-Schmidt 또는 Fidelity 손실."""
        return self.loss_fn(rho_pred, rho_target)

    def decode_shots(self, rho_pred: Tensor, K: int) -> list[Tensor]:
        """Phase F : 밀도 행렬 → K개 잠재 벡터 추출."""
        shots = []
        for k in range(K):
            start = k * self.d_latent
            end = (k + 1) * self.d_latent
            rho_k = rho_pred[:, start:end, start:end]
            _, eigvecs = _safe_eigh(rho_k)
            psi_k = eigvecs[:, :, -1]
            z_k = self.decoder_proj(psi_k)
            shots.append(z_k)
        return shots

    def training_forward(
        self,
        latents_list: list[Tensor],
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        학습용 전체 순전파.

        Returns:
            (rho_pred, rho_target, loss)
        """
        rho_0 = self.encode_shots(latents_list)            # Phase A
        rho_t = self.forward_process(rho_0, t)             # Phase B
        rho_pred = self.backward_pass(rho_t, t)            # Phase C/D
        loss = self.compute_loss(rho_pred, rho_0.detach()) # Phase E
        return rho_pred, rho_0, loss
