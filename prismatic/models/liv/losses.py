"""
LIV Contrastive Loss (L2)

InfoNCE 기반 contrastive loss. LIV가 task-relevant 물체 상태를 인코딩하고
로봇 팔 자세에 불변하도록 유도.

Positive 쌍 정의:
    같은 물체 상태 + 다른 팔 자세 (Isaac Sim에서 생성)
    → LIV가 팔 움직임에 불변하지 않으면 정상 실행 중 false-positive stop 발생

LIVModule의 ProjectionMLP가 이미 L2 정규화를 수행하므로
similarity = dot product (cosine similarity와 동일).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIVContrastiveLoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss for LIV training.

    in-batch negatives 방식: 같은 배치 내 다른 샘플이 자동으로 negative.
    symmetric loss: anchor→positive, positive→anchor 양방향 평균.

    Args:
        temperature: softmax temperature τ. 0.07~0.2 사이에서 튜닝.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D) — 현재 프레임 LIV (L2 normalized)
            positive: (B, D) — 같은 물체 상태, 다른 팔 자세의 LIV (L2 normalized)

        Returns:
            scalar loss
        """
        B = anchor.shape[0]

        # similarity matrix: sim[i, j] = anchor_i · positive_j / τ
        sim = torch.matmul(anchor, positive.T) / self.temperature   # (B, B)

        # diagonal = positive pair
        labels = torch.arange(B, device=anchor.device)

        # symmetric: anchor→positive & positive→anchor
        loss_a2p = F.cross_entropy(sim, labels)
        loss_p2a = F.cross_entropy(sim.T, labels)

        return (loss_a2p + loss_p2a) / 2
