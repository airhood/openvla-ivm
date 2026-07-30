"""
LIV Module (Latent Intent Vector)

LLM의 Action-to-Vision Attention과 Vision Token Feature를 결합하여
task-relevant 정보를 압축한 Latent Intent Vector를 추출하는 모듈.

수식 기반 Pipeline:
    (1) Action→Vision attention 추출 & action token mean:
            ᾱⱼ^{l,h} = (1/T) Σᵢ αᵢⱼ^{l,h},   Ā ∈ R^{L×H×N}

    (2) Spatial pooling (MLP 입력 축소용): N=256 → P²
            Ā_pool ∈ R^{L×H×P²}

    (3) Attention MLP — L×H 공동 가중치:
            w = MLP_w(flatten(Ā_pool)) ∈ R^{L×H}

    (4) 원본 attention으로 position별 가중합 + softmax:
            ã_j = Σ_{l,h} w_{l,h} ᾱⱼ^{l,h},   a = softmax(ã) ∈ Δ^{N-1}

    (5) Attention Pooling (vision token weighted sum):
            z = Σⱼ aⱼ vⱼ ∈ R^D

    (6) LIV Projection MLP + L2 정규화:
            ℓ = g_φ(z) / ‖g_φ(z)‖₂ ∈ R^{liv_dim}
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_action_vision_submatrix(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    action_start: int,
    action_end: int,
    n_extract_layers: int,
) -> torch.Tensor:
    """(1) 마지막 L개 레이어의 action→vision 서브행렬을 추출하고 action token mean을 낸다.

    LIVModule.forward()의 첫 단계와 동일한 연산을 독립 함수로 뺀 것 — VLA forward(7B, 비쌈)의
    출력을 캐싱할 때 이 함수의 결과(Ā, 캐싱 대상)만 저장해두면, 이후 LIVModule 학습 루프에서는
    raw attentions(레이어 전체, 매우 큼) 없이 forward_from_submatrix()로 바로 이어붙일 수 있다.

    Args:
        attentions:   LLM output_attentions 결과. tuple[Tensor], len=n_layers,
                      각 원소 (B, n_heads, seq, seq)
        vision_start, vision_end: vision token 구간
        action_start, action_end: action token 구간
        n_extract_layers: 마지막 몇 개 레이어를 쓸지 (L)

    Returns:
        Ā: (B, L, H, N) — action token에 대해 평균낸 action→vision attention
    """
    selected = attentions[-n_extract_layers:]  # L × (B, H, seq, seq)
    sub = torch.stack(
        [a[:, :, action_start:action_end, vision_start:vision_end] for a in selected],
        dim=1,
    )  # (B, L, H, A, N)
    return sub.mean(dim=3)  # (B, L, H, N) — action token mean


class AttentionMLP(nn.Module):
    """L×H 공동 가중치를 생성하는 MLP.

    레이어와 헤드를 독립적으로 보지 않고 jointly 처리하여
    레이어 간 상호작용을 포착.

    Args:
        input_dim:  flatten(Ā_pool) 차원 = L*H*P²
        hidden_dim: 중간 hidden 차원
        output_dim: L*H (레이어×헤드 가중치 수)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L*H*P²)
        Returns:
            (B, L*H)
        """
        return self.net(x)


class ProjectionMLP(nn.Module):
    """z → L2-normalized LIV.

    contrastive loss가 cosine similarity 기반이므로
    출력을 단위구로 사영.

    Args:
        input_dim:  llm_hidden_dim (D)
        hidden_dim: 중간 hidden 차원
        output_dim: liv_dim (128)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, output_dim)  — L2 정규화된 단위벡터
        """
        return F.normalize(self.net(x), dim=-1)


class LIVModule(nn.Module):
    """Latent Intent Vector 추출 모듈.

    Args:
        n_heads:             LLM attention head 수 (Llama-2-7B = 32)
        n_action_tokens:     action token 수 (LIBERO = 56)
        n_vision_tokens:     vision token 수 (256 고정)
        llm_hidden_dim:      LLM hidden state 차원 (Llama-2-7B = 4096)
        n_extract_layers:    사용할 마지막 L개 레이어 수
        pool_size:           spatial pooling 출력 크기 P (MLP 입력 축소용)
        attn_mlp_hidden_dim: AttentionMLP hidden 차원
        proj_hidden_dim:     ProjectionMLP hidden 차원
        liv_dim:             LIV 출력 차원 (단위구 위)
    """

    def __init__(
        self,
        n_heads: int = 32,
        n_action_tokens: int = 56,
        n_vision_tokens: int = 256,
        llm_hidden_dim: int = 4096,
        n_extract_layers: int = 4,
        pool_size: int = 8,
        attn_mlp_hidden_dim: int = 256,
        proj_hidden_dim: int = 1024,
        liv_dim: int = 128,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_action_tokens = n_action_tokens
        self.n_vision_tokens = n_vision_tokens
        self.llm_hidden_dim = llm_hidden_dim
        self.n_extract_layers = n_extract_layers
        self.pool_size = pool_size

        self.vision_grid = int(math.isqrt(n_vision_tokens))
        assert self.vision_grid ** 2 == n_vision_tokens, (
            f"n_vision_tokens={n_vision_tokens}이 정방형이 아님"
        )

        # MLP 입력: L*H*P²
        flat_dim = n_extract_layers * n_heads * pool_size * pool_size

        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.attn_mlp = AttentionMLP(flat_dim, attn_mlp_hidden_dim, n_extract_layers * n_heads)
        self.proj_mlp = ProjectionMLP(llm_hidden_dim, proj_hidden_dim, liv_dim)

    def forward(
        self,
        attentions: tuple,
        vision_features: torch.Tensor,
        vision_start: int,
        vision_end: int,
        action_start: int,
        action_end: int,
    ) -> torch.Tensor:
        """
        Args:
            attentions:      LLM output_attentions 결과.
                             tuple[Tensor], len=n_layers, 각 원소 (B, n_heads, seq, seq)
            vision_features: MLP Projector 출력 (DINOv2+SigLIP → LLM dim).
                             (B, n_vision_tokens, llm_hidden_dim)
            vision_start:    vision token 시작 인덱스 (보통 1)
            vision_end:      vision token 끝 인덱스   (보통 257)
            action_start:    action token 시작 인덱스 (seq_len - n_action_tokens)
            action_end:      action token 끝 인덱스   (seq_len)

        Returns:
            ℓ: (B, liv_dim)  — L2 정규화된 Latent Intent Vector
        """
        sub = extract_action_vision_submatrix(
            attentions, vision_start, vision_end, action_start, action_end, self.n_extract_layers
        )  # (B, L, H, N)
        return self.forward_from_submatrix(sub, vision_features)

    def forward_from_submatrix(self, sub: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """extract_action_vision_submatrix()의 출력(Ā, 캐시에서 로드 가능)부터 이어서 계산.

        forward()의 (2)~(6) 단계와 동일. 캐싱 파이프라인(research/data_generation/build_liv_cache.py)이
        저장한 Ā + vision_features를 그대로 여기 넣으면, VLA 7B forward 없이 LIVModule만 학습할 수 있다.

        Args:
            sub:             (B, L, H, N) — extract_action_vision_submatrix() 출력과 동일 shape
            vision_features: (B, N, D)

        Returns:
            ℓ: (B, liv_dim) — L2 정규화된 Latent Intent Vector
        """
        B = vision_features.shape[0]
        G = self.vision_grid    # 16
        P = self.pool_size      # 8
        L = self.n_extract_layers
        H = self.n_heads

        assert sub.shape[1] >= L, f"캐시된 레이어 수({sub.shape[1]})가 n_extract_layers({L})보다 적음"
        sub = sub[:, -L:]  # 캐시가 ablation 대비로 더 많은 레이어를 담고 있을 수 있으므로 마지막 L개만 사용

        # ── (2) Spatial pooling for MLP input: N=256 → P² ──
        sub_pool = sub.contiguous().reshape(B * L * H, 1, G, G)  # (B·L·H, 1, 16, 16)
        sub_pool = self.spatial_pool(sub_pool)                     # (B·L·H, 1, P, P)
        sub_pool = sub_pool.reshape(B, L * H * P * P)             # (B, L·H·P²)

        # ── (3) Attention MLP → L×H 공동 가중치 ──
        w = self.attn_mlp(sub_pool)        # (B, L·H)
        w = w.reshape(B, L, H)             # (B, L, H)

        # ── (4) 원본 attention으로 position별 가중합 + softmax ──
        # ã_j = Σ_{l,h} w_{l,h} * ᾱ_j^{l,h}
        attn_map = torch.einsum("blh,blhn->bn", w, sub)   # (B, N=256)
        attn_map = F.softmax(attn_map, dim=-1)              # normalize over positions

        # ── (5) Attention Pooling: z = Σⱼ aⱼ vⱼ ──
        # vision_features: (B, N, D)
        z = torch.einsum("bn,bnd->bd", attn_map, vision_features)  # (B, D)

        # ── (6) LIV Projection MLP + L2 정규화 ──
        liv = self.proj_mlp(z)  # (B, liv_dim), L2-normalized

        return liv
