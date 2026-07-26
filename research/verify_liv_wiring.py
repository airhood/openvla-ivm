"""
research/verify_liv_wiring.py

predict_action(liv_module=...) 연결이 실제로 동작하는지 검증하는 sanity check.
학습이 아니라 배선(wiring) 확인용 — LIVModule은 랜덤 초기화 상태 그대로 쓴다.

확인 항목:
1. vla.last_liv가 AttributeError 없이 생성되는가
2. shape이 (B, liv_dim)인가
3. LIVModule의 ProjectionMLP가 L2 정규화를 하므로 각 벡터의 norm이 1에 가까운가
4. NaN/Inf 없이 finite한가

주의 — 알려진 제약: LIVModule은 현재 단일 이미지(정사각 vision token, 16x16=256)만
가정하고 있어서 (vision_grid = sqrt(n_vision_tokens) assert), num_images_in_input>1
(wrist_image 포함)이면 vision token이 256의 배수(예: 512)가 되어 LIVModule 생성자의
정사각형 assert에 걸린다. 그래서 이 검증은 --num_images_in_input 1(주 시점 카메라만)로
돌린다. 멀티 이미지 지원은 LIVModule 자체의 설계를 바꿔야 하는 별도 작업이며,
이 스크립트가 발견한 TODO로 남겨둔다 (docs/MODEL.md에 반영 필요).

사용:
    python research/verify_liv_wiring.py \
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key, initialize_model  # noqa: E402
from phase0_attention_heatmap import resolve_local_checkpoint  # noqa: E402
from prismatic.models.liv import LIVModule  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument(
        "--sample_pkl",
        type=str,
        default=str(REPO_ROOT / "experiments/robot/libero/sample_libero_spatial_observation.pkl"),
    )
    parser.add_argument("--liv_dim", type=int, default=128)
    parser.add_argument("--n_extract_layers", type=int, default=4)
    args = parser.parse_args()

    local_checkpoint = resolve_local_checkpoint(args.pretrained_checkpoint)

    cfg = GenerateConfig(
        pretrained_checkpoint=local_checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=True,
        use_proprio=True,
        use_film=False,
        num_images_in_input=1,  # LIVModule의 단일 이미지(정사각) 가정과 맞추기 위함. 위 docstring 참고.
        center_crop=True,
    )

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    check_unnorm_key(cfg, model)

    liv_module = LIVModule(
        n_heads=32,
        n_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
        n_vision_tokens=256,
        llm_hidden_dim=model.llm_dim,
        n_extract_layers=args.n_extract_layers,
        liv_dim=args.liv_dim,
    ).to(model.device, dtype=torch.bfloat16)

    print(f"Loading sample observation from {args.sample_pkl} ...")
    import pickle

    with open(args.sample_pkl, "rb") as f:
        sample = pickle.load(f)
    observation = {"full_image": sample["full_image"], "state": sample["state"]}
    task_description = sample["task_description"]

    print("Calling predict_action(..., liv_module=liv_module) ...")
    with torch.inference_mode():
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla

        primary_image = prepare_images_for_vla([observation["full_image"]], cfg)[0]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(observation["state"], proprio_norm_stats)

        model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=cfg.use_film,
            liv_module=liv_module,
        )

    # === 검증 ===
    assert hasattr(model, "last_liv"), "FAIL: vla.last_liv가 생성되지 않음 (liv_module 배선 문제)"
    liv = model.last_liv
    print(f"last_liv.shape: {tuple(liv.shape)}")
    assert liv.shape == (1, args.liv_dim), f"FAIL: shape 불일치 {tuple(liv.shape)} != (1, {args.liv_dim})"

    assert torch.isfinite(liv).all(), "FAIL: last_liv에 NaN/Inf 있음"

    norm = liv.float().norm(dim=-1)
    print(f"last_liv L2 norm: {norm.item():.4f}")
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-2), f"FAIL: L2 정규화 안 됨 (norm={norm.item():.4f})"

    print("PASS — LIVModule이 predict_action()에 정상적으로 배선되어 있음")


if __name__ == "__main__":
    main()
