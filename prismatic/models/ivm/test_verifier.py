import importlib.util

import torch

spec = importlib.util.spec_from_file_location("verifier", "prismatic/models/ivm/verifier.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MLPVisionBackbone = mod.MLPVisionBackbone
MetricIVM = mod.MetricIVM

spec2 = importlib.util.spec_from_file_location("losses", "prismatic/models/liv/losses.py")
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
LIVContrastiveLoss = mod2.LIVContrastiveLoss

B, D, K = 4, 128, 3
backbone = MLPVisionBackbone(resize_to=64, output_dim=D)
ivm = MetricIVM(backbone)

liv = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)  # ℓ, L2-normalized처럼
valid_img = torch.rand(B, 3, 256, 256)
invalid_imgs = torch.rand(B, K, 3, 256, 256)

e_valid = ivm.embed(valid_img)
print("e_valid.shape:", tuple(e_valid.shape))
assert e_valid.shape == (B, D), "FAIL: valid embedding shape"

e_invalid = ivm.embed(invalid_imgs.reshape(B * K, 3, 256, 256)).reshape(B, K, D)
print("e_invalid.shape:", tuple(e_invalid.shape))
assert e_invalid.shape == (B, K, D), "FAIL: invalid embedding shape"

norm = e_valid.norm(dim=-1)
print("e_valid L2 norm:", norm)
assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), "FAIL: L2 정규화 안 됨"

sim = ivm(liv, valid_img)
print("cos(liv, e_valid).shape:", tuple(sim.shape))
assert sim.shape == (B,), "FAIL: forward() 출력 shape"

# L2와 동일한 InfoNCE loss 재사용 확인 (liv=anchor, valid=positive, invalid=hard negative)
loss_fn = LIVContrastiveLoss(temperature=0.07)
loss = loss_fn(liv, e_valid, e_invalid)
print("IVM contrastive loss (LIVContrastiveLoss 재사용):", loss.item())
assert torch.isfinite(loss), "FAIL: loss가 NaN/Inf"
loss.backward()
assert backbone.net[0].weight.grad is not None, "FAIL: backbone에 gradient 안 흐름"

print("PASS — MetricIVM + MLPVisionBackbone shape/gradient/loss-재사용 전부 정상")
