"""
research/phase0_attention_heatmap.py

Phase 0: Attention heatmap 검증 (학습 없음) — docs/MODEL.md §6, §8 Phase 0.

사전학습된 OFT 체크포인트로 LIBERO 프레임 하나를 forward하고,
action→vision attention이 조작 대상 물체에 집중하는 head/layer가
존재하는지 시각적으로 확인한다. LIV 설계 전체의 전제를 검증하는 실험이라
LIVModule 학습(Phase 1) 이전에 가장 먼저 돌려야 한다.

여기서 보는 attention은 학습된 AttentionMLP로 가중합하기 전의 RAW attention이다
(LIVModule이 아직 없으므로). "집중하는 head/layer가 하나라도 있는가"만 확인한다.

실행 환경: Colab/Ubuntu (GPU, LIBERO 설치 필요). 로컬 CPU-only 환경에서는
7B 체크포인트 로드가 불가능하므로 이 스크립트는 여기서 실행할 수 없다.

사용:
    python research/phase0_attention_heatmap.py \
        --pretrained_checkpoint <path or HF repo id> \
        --task_suite_name libero_spatial \
        --task_id 0 \
        --episode_idx 0 \
        --n_extract_layers 4 \
        --output_dir ./research/phase0_out
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# MuJoCo/robosuite import보다 먼저 설정해야 함 — 헤드리스(Colab 등, 디스플레이 없음) 환경에서
# 렌더 백엔드를 못 잡으면 OffScreenRenderEnv가 Python 예외 없이 세그폴트로 죽는다.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

# Repo root를 sys.path에 추가 (실행 위치와 무관하게 동작하도록)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    check_unnorm_key,
    initialize_model,
    prepare_observation,
)
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def resolve_local_checkpoint(checkpoint: str) -> str:
    """HF Hub 체크포인트 ID면 로컬로 다운로드해서 그 경로를 반환한다.

    `AutoModelForVision2Seq.from_pretrained(hub_id, trust_remote_code=True)`는 체크포인트에
    번들된 원본 modeling_prismatic.py를 그대로 쓰고, 로컬에서 수정한 버전(output_attentions
    지원 등)은 반영되지 않는다. `get_vla()`의 `check_model_logic_mismatch()`/`update_auto_map()`가
    로컬 코드를 체크포인트 디렉토리에 동기화해주는 메커니즘인데, 이건 로컬 디렉토리 경로에서만
    동작하고 Hub ID 문자열로는 동작하지 않는다. 그래서 먼저 로컬로 받아온다.
    """
    if os.path.isdir(checkpoint):
        return checkpoint
    local_dir = REPO_ROOT / "_checkpoints" / checkpoint.replace("/", "__")
    print(f"Downloading checkpoint {checkpoint} to {local_dir} (so local modeling code changes take effect)...")
    snapshot_download(repo_id=checkpoint, local_dir=str(local_dir))
    return str(local_dir)


def get_action_and_attention(cfg, vla, processor, obs, task_label, action_head, proprio_projector, use_film):
    """experiments/robot/openvla_utils.py의 get_vla_action()과 동일한 입력 준비 로직이지만
    output_attentions=True로 predict_action을 호출하고 raw attention을 함께 반환한다.
    """
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
        all_images = prepare_images_for_vla(all_images, cfg)

        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(obs["state"], proprio_norm_stats)

        action, _ = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=use_film,
            output_attentions=True,
        )

    # predict_action()이 output_attentions=True일 때 stash해두는 값 (modeling_prismatic.py 참고)
    attentions = vla.last_attentions  # tuple(len=n_layers) of (1, n_heads, seq, seq)
    layout = vla.last_attention_layout  # {"vision_start", "vision_end", "action_start", "action_end"}
    return action, attentions, layout


def build_raw_heatmaps(attentions, layout, n_extract_layers: int = 4, image_idx: int = 0):
    """RAW action→vision attention을 (layer, head)별 16x16 heatmap으로 변환.

    LIVModule의 AttentionMLP 가중합을 거치지 않은 순수 attention이다 —
    Phase 0의 목적은 "학습 없이도 집중하는 head/layer가 존재하는가"를 보는 것.

    vision token 블록은 이미지당 256개(16x16, SigLIP/DINOv2 고정)씩 이어붙어 있다
    (num_images_in_input>1이면 [primary 256개, wrist 256개, ...] 순서). image_idx로
    시각화할 이미지를 선택한다 (기본 0 = primary/full_image, raw_img와 대응).

    Returns:
        heatmaps: (n_extract_layers, n_heads, grid, grid) numpy array, 각 (layer,head)별 0~1 정규화
        mean_heatmap: (grid, grid) — 선택된 모든 layer/head 평균
    """
    PATCHES_PER_IMAGE = 256  # SigLIP/DINOv2 비전 백본 고정값 (16x16 패치)
    grid = int(round(PATCHES_PER_IMAGE**0.5))  # 16

    vs, ve = layout["vision_start"], layout["vision_end"]
    as_, ae = layout["action_start"], layout["action_end"]
    n_vision = ve - vs
    assert n_vision % PATCHES_PER_IMAGE == 0, f"vision token 수({n_vision})가 {PATCHES_PER_IMAGE}의 배수가 아님"
    n_images = n_vision // PATCHES_PER_IMAGE
    assert 0 <= image_idx < n_images, f"image_idx={image_idx}가 이미지 수({n_images}) 범위를 벗어남"

    selected = attentions[-n_extract_layers:]  # L x (1, H, seq, seq)
    sub = torch.stack(
        [a[:, :, as_:ae, vs:ve] for a in selected], dim=0
    )  # (L, 1, H, A, N)
    sub = sub.squeeze(1)  # (L, H, A, N)
    sub = sub.mean(dim=2)  # action mean -> (L, H, N)
    sub = sub.reshape(sub.shape[0], sub.shape[1], n_images, grid, grid)  # (L, H, n_images, grid, grid)
    sub = sub[:, :, image_idx]  # 시각화할 이미지 선택 -> (L, H, grid, grid)

    heatmaps = sub.float().cpu().numpy()
    # per-(layer,head) min-max 정규화 (시각화용)
    flat = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], -1)
    mins = flat.min(axis=-1, keepdims=True)
    maxs = flat.max(axis=-1, keepdims=True)
    heatmaps_norm = (flat - mins) / np.clip(maxs - mins, 1e-8, None)
    heatmaps_norm = heatmaps_norm.reshape(heatmaps.shape)

    mean_heatmap = heatmaps_norm.mean(axis=(0, 1))  # (grid, grid)

    return heatmaps_norm, mean_heatmap


def save_visualization(image: np.ndarray, heatmaps: np.ndarray, mean_heatmap: np.ndarray, output_path: Path):
    """원본 이미지 + layer/head별 heatmap grid + 평균 heatmap을 하나의 figure로 저장."""
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    n_layers, n_heads = heatmaps.shape[:2]
    img_res = image.shape[0]

    def upsample(hm):
        pil = PILImage.fromarray((hm * 255).astype(np.uint8))
        pil = pil.resize((img_res, img_res), resample=PILImage.BILINEAR)
        return np.array(pil) / 255.0

    fig, axes = plt.subplots(n_layers + 1, n_heads + 1, figsize=(2 * (n_heads + 1), 2 * (n_layers + 1)))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("input")
    axes[0, 0].axis("off")

    for h in range(n_heads):
        axes[0, h + 1].imshow(image)
        axes[0, h + 1].imshow(upsample(heatmaps[-1, h]), cmap="jet", alpha=0.5)
        axes[0, h + 1].set_title(f"head {h}")
        axes[0, h + 1].axis("off")

    for l in range(n_layers):
        axes[l + 1, 0].imshow(image)
        axes[l + 1, 0].imshow(upsample(heatmaps[l].mean(axis=0)), cmap="jet", alpha=0.5)
        axes[l + 1, 0].set_title(f"layer -{n_layers - l} (head mean)")
        axes[l + 1, 0].axis("off")
        for h in range(n_heads):
            axes[l + 1, h + 1].imshow(image)
            axes[l + 1, h + 1].imshow(upsample(heatmaps[l, h]), cmap="jet", alpha=0.5)
            axes[l + 1, h + 1].axis("off")

    fig.suptitle("Action -> Vision attention (row: layer, col: head). Top-left row/col = summary.")
    fig.tight_layout()
    fig.savefig(output_path / "heatmap_grid.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
    ax2[0].imshow(image)
    ax2[0].set_title("input")
    ax2[0].axis("off")
    ax2[1].imshow(image)
    ax2[1].imshow(upsample(mean_heatmap), cmap="jet", alpha=0.5)
    ax2[1].set_title(f"mean over last {n_layers if (n_layers := heatmaps.shape[0]) else ''} layers x all heads")
    ax2[1].axis("off")
    fig2.tight_layout()
    fig2.savefig(output_path / "heatmap_mean.png", dpi=120)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--n_extract_layers", type=int, default=4)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--use_l1_regression", action="store_true", default=True)
    parser.add_argument("--use_proprio", action="store_true", default=True)
    parser.add_argument("--use_film", action="store_true", default=False)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./research/phase0_out")
    parser.add_argument(
        "--use_live_env",
        action="store_true",
        default=False,
        help="LIBERO 시뮬레이터를 실제로 띄워서 렌더링 (헤드리스 환경에서 렌더 백엔드 문제 발생 가능). "
        "기본값은 repo에 이미 포함된 사전 렌더링 관측값(sample_libero_spatial_observation.pkl) 사용.",
    )
    parser.add_argument(
        "--sample_pkl",
        type=str,
        default=str(REPO_ROOT / "experiments/robot/libero/sample_libero_spatial_observation.pkl"),
        help="--use_live_env 미사용 시 로드할 사전 렌더링 관측값 pickle 경로",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_checkpoint = resolve_local_checkpoint(args.pretrained_checkpoint)

    cfg = GenerateConfig(
        pretrained_checkpoint=local_checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=args.use_l1_regression,
        use_proprio=args.use_proprio,
        use_film=args.use_film,
        num_images_in_input=args.num_images_in_input,
        center_crop=args.center_crop,
        num_steps_wait=args.num_steps_wait,
    )

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    if args.use_live_env:
        # 라이브 시뮬레이터 경로: LIBERO env를 실제로 띄워서 렌더링.
        # 헤드리스 환경(Colab 등)에서는 MuJoCo 렌더 백엔드(EGL/OSMesa/GLFW) 문제로 죽을 수 있음.
        print(f"Setting up LIBERO env: {cfg.task_suite_name} / task {args.task_id}")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        task = task_suite.get_task(args.task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        initial_states = task_suite.get_task_init_states(args.task_id)
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])

        # 물체가 안정화될 때까지 잠깐 대기 (run_libero_eval.py와 동일 패턴)
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        observation, raw_img = prepare_observation(obs, resize_size)
    else:
        # 기본 경로: repo에 포함된 사전 렌더링 관측값 사용 (라이브 시뮬레이터 렌더링 불필요).
        # Phase 0은 이미지 1장에 대한 attention 패턴만 보면 되므로, 실시간 rollout이 필요 없다.
        print(f"Loading sample observation from {args.sample_pkl} ...")
        with open(args.sample_pkl, "rb") as f:
            sample = pickle.load(f)
        observation = {
            "full_image": sample["full_image"],
            "wrist_image": sample["wrist_image"],
            "state": sample["state"],
        }
        raw_img = sample["full_image"]
        task_description = sample["task_description"]

    print(f"Task: {task_description}")
    print("Running forward pass with output_attentions=True ...")
    action, attentions, layout = get_action_and_attention(
        cfg, model, processor, observation, task_description, action_head, proprio_projector, cfg.use_film
    )

    print(f"Predicted action[0]: {action[0]}")
    print(f"Attention layout: {layout}")
    print(f"n_layers total: {len(attentions)}, attn shape per layer: {tuple(attentions[0].shape)}")

    heatmaps, mean_heatmap = build_raw_heatmaps(attentions, layout, n_extract_layers=args.n_extract_layers)
    save_visualization(raw_img, heatmaps, mean_heatmap, output_dir)

    print(f"Saved heatmap_grid.png / heatmap_mean.png to {output_dir}")
    print(
        "판단 기준: heatmap_mean.png에서 밝은 영역이 조작 대상 물체에 걸쳐 있으면 통과. "
        "heatmap_grid.png에서 개별 head 중 하나라도 물체에 집중하면 통과 (AttentionMLP가 그 head를 골라내면 됨). "
        "전부 배경/구석/대각선 등 diffuse하면 MODEL.md §9의 hidden-state fallback 검토."
    )


if __name__ == "__main__":
    main()
