import torch
import torch.nn.functional as F

from losses import LIVContrastiveLoss

torch.manual_seed(0)

B, D, K = 8, 128, 4
anchor = F.normalize(torch.randn(B, D), dim=-1)
positive = F.normalize(torch.randn(B, D), dim=-1)
hard_negatives = F.normalize(torch.randn(B, K, D), dim=-1)

loss_fn = LIVContrastiveLoss(temperature=0.07)

# 1) hard negative 없이 (기존 동작, 하위 호환)
loss_no_hard = loss_fn(anchor, positive)
print("loss (no hard negatives):", loss_no_hard.item())
assert loss_no_hard.dim() == 0 and torch.isfinite(loss_no_hard)

# 2) hard negative 포함
loss_with_hard = loss_fn(anchor, positive, hard_negatives)
print("loss (with hard negatives):", loss_with_hard.item())
assert loss_with_hard.dim() == 0 and torch.isfinite(loss_with_hard)

# 3) anchor == positive (완벽한 매칭)면 loss가 낮아야 함
perfect_anchor = F.normalize(torch.randn(B, D), dim=-1)
perfect_positive = perfect_anchor.clone()
loss_perfect = loss_fn(perfect_anchor, perfect_positive)
print("loss (perfect match, no hard negatives):", loss_perfect.item())
assert loss_perfect.item() < loss_no_hard.item()

# 4) gradient가 anchor/positive/hard_negatives 전부로 흐르는지 확인
anchor_g = F.normalize(torch.randn(B, D), dim=-1).requires_grad_()
positive_g = F.normalize(torch.randn(B, D), dim=-1).requires_grad_()
hard_g = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
loss_g = loss_fn(anchor_g, positive_g, hard_g)
loss_g.backward()
assert anchor_g.grad is not None and torch.isfinite(anchor_g.grad).all()
assert positive_g.grad is not None and torch.isfinite(positive_g.grad).all()
assert hard_g.grad is not None and torch.isfinite(hard_g.grad).all()

print("PASS")
