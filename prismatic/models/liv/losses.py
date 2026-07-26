"""
LIV Contrastive Loss (L2)

InfoNCE 기반 contrastive loss. LIV가 task-relevant 물체 상태를 인코딩하고
로봇 팔 자세에 불변하도록 유도.

Positive 쌍 정의:
    같은 물체 상태 + 다른 팔 자세 (robosuite/MuJoCo에서 생성)
    → LIV가 팔 움직임에 불변하지 않으면 정상 실행 중 false-positive stop 발생

Negative 정의 (두 종류를 함께 씀 — 하나만 쓰면 문제가 생김):
    - in-batch negative: 배치 내 다른 샘플 (대개 다른 장면/태스크)
    - hard negative: 같은 장면 + 물체만 교란 (이동/회전/제거/교체)

    in-batch negative만 쓰면 "이 장면 vs 저 장면" 구분, 즉 배경만 보고도 풀리는
    쉬운 문제가 되어버려서 LIV가 물체의 미세한 상태 변화를 인코딩할 유인이 없어짐.
    hard negative(배경은 같고 물체만 다름)가 있어야 물체 상태를 실제로 봐야만
    구분이 되고, 그게 우리가 원하는 신호임.

LIVModule의 ProjectionMLP가 이미 L2 정규화를 수행하므로
similarity = dot product (cosine similarity와 동일).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIVContrastiveLoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss for LIV training, in-batch + hard negative 결합.

    symmetric loss: anchor→positive, positive→anchor 양방향 평균.
    hard negative는 anchor→positive 방향에만 추가됨 (각 anchor에 종속된 negative라
    positive→anchor 방향의 "후보 집합"에는 자연스럽게 속하지 않음).

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
        hard_negatives: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor:         (B, D) — 현재 프레임 LIV (L2 normalized)
            positive:       (B, D) — 같은 물체 상태, 다른 팔 자세의 LIV (L2 normalized)
            hard_negatives: (B, K, D) 또는 None — anchor_i와 같은 장면에서 물체만 교란한
                            K개의 LIV (L2 normalized). None이면 in-batch negative만 사용.

        Returns:
            scalar loss
        """
        B = anchor.shape[0]

        # in-batch similarity: sim[i, j] = anchor_i · positive_j / τ  (대각선 = positive pair)
        sim_inbatch = torch.matmul(anchor, positive.T) / self.temperature  # (B, B)

        if hard_negatives is not None:
            # anchor_i · hard_negatives_i_k / τ  (각 anchor 고유의 negative)
            sim_hard = torch.einsum("bd,bkd->bk", anchor, hard_negatives) / self.temperature  # (B, K)
            logits_a2p = torch.cat([sim_inbatch, sim_hard], dim=1)  # (B, B+K)
        else:
            logits_a2p = sim_inbatch

        labels = torch.arange(B, device=anchor.device)

        # symmetric: anchor→positive(+hard negative) & positive→anchor(in-batch만)
        loss_a2p = F.cross_entropy(logits_a2p, labels)
        loss_p2a = F.cross_entropy(sim_inbatch.T, labels)

        return (loss_a2p + loss_p2a) / 2
