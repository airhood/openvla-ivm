"""
prismatic/models/ivm/verifier.py

IVM (Intent Verification Module) - metric 기반 검증기 (docs/MODEL.md §4.2, 첫 프로토타입).

VLA/LIV와 완전히 분리된, 로봇 온보드에서 도는 경량 vision network. 서버에서 전송받은 ℓ(LIV)와
같은 공간(R^d)으로 현재 관측 이미지를 임베딩해서 코사인 유사도로 valid/invalid를 판정한다:

    e = f_ψ(I′) → R^d (LIV와 같은 공간)
    validity = cos(ℓ, e) > threshold

FiLM 분류기(§4.1, main)는 나중 단계 — 이 모듈은 backbone 후보 사다리(MLP → ResNet-18 →
MobileViT)의 첫 단계인 MLP baseline만 구현한다. MLP는 원본 해상도가 아니라 작게 리사이즈한
이미지를 flatten해서 씀 — ResNet-18과 대비되는 "컨볼루션(공간 구조) 유무" ablation 축이라
순수 MLP(공간 구조 없음)로 구현.

학습은 L2와 동일한 InfoNCE 구조를 그대로 씀(docs/MODEL.md §4.2 "IVM 학습도 contrastive") —
prismatic.models.liv.LIVContrastiveLoss를 ℓ=anchor, valid 임베딩=positive, invalid 임베딩=
hard negative로 놓고 그대로 재사용한다(별도 loss 클래스 불필요).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVisionBackbone(nn.Module):
    """IVM의 f_ψ MLP baseline. 이미지를 (resize_to, resize_to)로 줄인 뒤 flatten+MLP로 R^d에 임베딩.

    Args:
        resize_to: 입력 이미지를 이 정사각형 해상도로 리사이즈(기본 64) — 원본(256)을 그대로 flatten하면
                   입력 차원이 너무 커짐(196608) + "MLP baseline"은 공간 구조를 활용 안 하는 게
                   ResNet-18/MobileViT 대비 ablation 취지에 맞음
        in_channels: 입력 채널 수 (RGB=3)
        hidden_dim: MLP hidden 차원
        output_dim: 출력 임베딩 차원 (LIV와 같은 공간이어야 함, 기본 128)
    """

    def __init__(self, resize_to: int = 64, in_channels: int = 3, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.resize_to = resize_to
        input_dim = in_channels * resize_to * resize_to
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) float — 0~1 정규화는 호출부 책임
        Returns:
            (B, output_dim) — L2 정규화된 임베딩 (LIV와 코사인 유사도로 직접 비교 가능)
        """
        if image.shape[-2:] != (self.resize_to, self.resize_to):
            image = F.interpolate(image, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        flat = image.reshape(image.shape[0], -1)
        return F.normalize(self.net(flat), dim=-1)


class MetricIVM(nn.Module):
    """metric 기반 IVM 전체. backbone은 교체 가능(현재 MLPVisionBackbone만 구현).

    Args:
        backbone: f_ψ 역할을 하는 nn.Module (image -> (B, liv_dim) L2-normalized 임베딩)
        threshold: cos(ℓ, e) > threshold면 valid. 배포 후 재학습 없이 조정 가능(§4.2 장점)
    """

    def __init__(self, backbone: nn.Module, threshold: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.threshold = threshold

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, liv: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            liv:   (B, liv_dim) — 서버에서 받은 ℓ (L2 정규화됨)
            image: (B, C, H, W) — 현재 온보드 관측
        Returns:
            (B,) — cosine similarity cos(ℓ, e). 호출부에서 `> threshold`로 valid 판정
        """
        e = self.embed(image)
        return torch.sum(liv * e, dim=-1)
