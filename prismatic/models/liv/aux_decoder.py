"""
L2b — LIV 보조 supervision (Auxiliary Object-State Decoder)

LIV(ℓ)를 contrastive(L2) 하나로만 학습하면, 물체 상태를 실제로 인코딩하지 않고도
손실을 낮추는 지름길(배경 등)이 있을 수 있다. 이를 막기 위해 LIV에서 조작 대상
물체의 위치/자세를 직접 예측하도록 시키는 보조 디코더 + 손실.

학습 전용: 추론(실제 로봇 동작) 시에는 사용하지 않고 버린다. LIV → IVM 경로만 남는다.
Dec 예측 오차 자체가 LIV 품질의 정량 지표가 된다 (t-SNE보다 명확한 검증 수단).

물체 상태 표현: position(3) + quaternion(4, MuJoCo 관례상 (w,x,y,z) 순서) = 7차원.
quaternion은 q와 -q가 같은 회전을 나타내는 double-cover 문제가 있어서, 정규화 없이
MSE를 그대로 쓰면 부호 모호성 때문에 학습이 불안정해질 수 있다. 데이터 파이프라인에서
GT quaternion을 canonicalize_quaternion()으로 w>=0 반구로 정렬해서 넘겨야 한다.
"""

import torch
import torch.nn as nn


class ObjectStateDecoder(nn.Module):
    """LIV → 물체 위치/자세 예측. 학습 전용, 추론 시 버림.

    Args:
        liv_dim: 입력 LIV 차원
        hidden_dim: MLP hidden 차원
        state_dim: 출력 차원 (기본 7 = position(3) + quaternion(4))
    """

    def __init__(self, liv_dim: int = 128, hidden_dim: int = 256, state_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(liv_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, liv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            liv: (B, liv_dim)
        Returns:
            (B, state_dim) — 예측된 물체 위치/자세 (canonicalize된 GT와 동일 규약)
        """
        return self.net(liv)


class ObjectStateLoss(nn.Module):
    """L2b: 예측된 물체 상태와 GT 사이의 MSE.

    GT의 quaternion 성분은 호출 전에 canonicalize_quaternion()으로 w>=0 정규화되어
    있어야 한다 — 그렇지 않으면 같은 회전이 부호만 다른 두 벡터로 나타나 손실이
    불필요하게 커지고 학습이 불안정해진다.
    """

    def forward(self, predicted_state: torch.Tensor, gt_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_state: (B, state_dim) — ObjectStateDecoder 출력
            gt_state:        (B, state_dim) — 시뮬레이터 GT (position + canonicalized quaternion)

        Returns:
            scalar loss
        """
        return torch.mean((predicted_state - gt_state) ** 2)


def canonicalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """quaternion을 w>=0 반구로 정규화한다 (q와 -q가 같은 회전을 나타내는 이중 표현 문제 해소).

    Args:
        quat: (..., 4) — (w, x, y, z) 순서 (MuJoCo 관례)

    Returns:
        (..., 4) — w>=0으로 부호 정렬된 quaternion
    """
    sign = torch.where(quat[..., 0:1] < 0, -1.0, 1.0)
    return quat * sign
