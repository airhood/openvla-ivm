import torch
import torch.nn.functional as F

from aux_decoder import ObjectStateDecoder, ObjectStateLoss, canonicalize_quaternion

torch.manual_seed(0)

B, LIV_DIM, STATE_DIM = 8, 128, 7

# 1) shape 검증
decoder = ObjectStateDecoder(liv_dim=LIV_DIM, state_dim=STATE_DIM)
liv = F.normalize(torch.randn(B, LIV_DIM), dim=-1)
pred = decoder(liv)
print("pred.shape:", pred.shape)
assert pred.shape == (B, STATE_DIM)

# 2) canonicalize_quaternion: w<0인 quaternion은 부호가 뒤집혀야 함
quat = torch.tensor([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, -0.5]])
canon = canonicalize_quaternion(quat)
print("canonicalized:", canon)
assert torch.all(canon[:, 0] >= 0)
assert torch.allclose(canon[0], -quat[0])  # w<0 -> 부호 반전
assert torch.allclose(canon[1], quat[1])   # 이미 w>=0 -> 그대로

# 3) q와 -q가 canonicalize 후 동일해지는지 (double-cover 해소 확인)
q = F.normalize(torch.randn(4), dim=0)
neg_q = -q
assert torch.allclose(canonicalize_quaternion(q), canonicalize_quaternion(neg_q))

# 4) perfect prediction -> loss ~0
loss_fn = ObjectStateLoss()
gt = torch.randn(B, STATE_DIM)
loss_perfect = loss_fn(gt, gt)
print("loss (perfect):", loss_perfect.item())
assert loss_perfect.item() < 1e-6

# 5) 실제 예측 vs GT loss는 0보다 커야 함 + gradient 흐름 확인
liv_g = F.normalize(torch.randn(B, LIV_DIM), dim=-1).requires_grad_()
pred_g = decoder(liv_g)
gt_random = torch.randn(B, STATE_DIM)
loss = loss_fn(pred_g, gt_random)
print("loss (random):", loss.item())
assert loss.item() > 0
loss.backward()
assert liv_g.grad is not None and torch.isfinite(liv_g.grad).all()
for p in decoder.parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all()

print("PASS")
