import importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    "liv_module", "prismatic/models/liv/liv_module.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
LIVModule = mod.LIVModule

B, D, seq = 2, 4096, 344
model = LIVModule(llm_hidden_dim=D)

dummy_attn = tuple(torch.randn(B, 32, seq, seq) for _ in range(32))
dummy_vis = torch.randn(B, 256, D)

liv = model(dummy_attn, dummy_vis, 1, 257, 288, 344)
print("liv.shape:", liv.shape)
print("L2 norm:", liv.norm(dim=-1))
print("PASS" if liv.shape == (B, 128) else "FAIL")
