"""
research/data_generation/build_liv_cache.py

generate_dataset.py가 만든 manifest.jsonl + 이미지를 읽어서, OFT 체크포인트로 각 이미지를
forward(output_attentions=True)한 뒤 action→vision attention 서브행렬(Ā)과 vision token feature를
추출해 디스크에 캐싱한다. Phase 1(LIV 학습) 스텝 개요(docs/MODEL.md §6)의 "VLA forward" 부분을
한 번만 수행해두면, 이후 LIVModule 학습 루프는 7B forward 없이(LIVModule.forward_from_submatrix()로)
이 캐시만 반복해서 읽어 학습할 수 있다.

GPU 필요(7B 모델) — Colab/A100에서 실행. generate_dataset.py(로컬 Ubuntu, robosuite 렌더링)의
출력을 입력으로 받으며, 이 스크립트 자체는 robosuite/LIBERO 라이브 env를 띄우지 않는다
(이미지는 이미 전부 PNG로 저장돼 있음) — 단, run_libero_eval.py를 통해 import되므로 LIBERO
패키지 설치 자체는 필요하다(실행되는 함수만 env-free).

설계 결정 — proprio 미포함: scene_utils 기반 합성 이미지에는 실제 로봇 proprio(EE pose 등)가
없다. 값을 0으로 채워 넣으면 정규화 공간에서 "0"이 임의의 편향으로 attention에 영향을 줄 수
있으므로, 대신 cfg.use_proprio=False로 proprio 토큰 자체를 시퀀스에서 뺀다(모델이 지원하는
정상 경로 — predict_action은 proprio_projector가 None이면 자동으로 proprio를 건너뜀).

캐싱되는 레이어 수(--n_extract_layers, 기본 8)는 LIVModule 학습 시 쓸 n_extract_layers(기본 4,
docs/MODEL.md §3.3 ablation 후보 1/4/8)보다 크거나 같아야 한다 — 캐시를 다시 만들지 않고도
여러 n_extract_layers 값으로 ablation할 수 있도록 여유 있게 캐싱해둔다
(LIVModule.forward_from_submatrix()가 마지막 L개만 슬라이싱해서 사용).

LIVModule의 단일-이미지 전제(docs/MODEL.md §3.3.1)에 맞춰 num_images_in_input=1로 고정.

사용:
    python research/data_generation/build_liv_cache.py \
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --manifest research/data_generation/dataset_out/manifest.jsonl \
        --output_dir research/data_generation/liv_cache_out
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # research/ (resolve_local_checkpoint)

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key, initialize_model  # noqa: E402
from experiments.robot.openvla_utils import DEVICE, prepare_images_for_vla  # noqa: E402
from phase0_attention_heatmap import resolve_local_checkpoint  # noqa: E402
from prismatic.models.liv import extract_action_vision_submatrix  # noqa: E402


def extract_sample(model, processor, cfg, action_head, image: np.ndarray, task_description: str, n_extract_layers: int):
    """이미지 1장을 forward(output_attentions=True)하고 (submatrix, vision_features)를 numpy로 반환.

    Returns:
        submatrix:       (n_extract_layers, n_heads, n_vision_tokens) float16
        vision_features:  (n_vision_tokens, llm_hidden_dim) float16
    """
    with torch.inference_mode():
        primary_image = prepare_images_for_vla([image], cfg)[0]
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=None,
            proprio_projector=None,
            action_head=action_head,
            use_film=cfg.use_film,
            output_attentions=True,
        )

    attentions = model.last_attentions
    layout = model.last_attention_layout
    vision_features = model.last_pure_vision_features  # (1, n_vision_tokens, D)

    sub = extract_action_vision_submatrix(
        attentions,
        layout["vision_start"],
        layout["vision_end"],
        layout["action_start"],
        layout["action_end"],
        n_extract_layers,
    )  # (1, L, H, N)

    sub = sub.squeeze(0).to(torch.float16).cpu().numpy()
    vision_features = vision_features.squeeze(0).to(torch.float16).cpu().numpy()
    return sub, vision_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True, help="generate_dataset.py가 만든 manifest.jsonl 경로")
    parser.add_argument(
        "--output_dir", type=str, default=str(REPO_ROOT / "research/data_generation/liv_cache_out")
    )
    parser.add_argument(
        "--n_extract_layers",
        type=int,
        default=8,
        help="캐싱할 마지막 레이어 수 (ablation 후보 중 최댓값 이상 권장, 기본 8)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataset_dir = manifest_path.parent  # image_path가 이 기준 상대경로
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} rows from {manifest_path}")

    local_checkpoint = resolve_local_checkpoint(args.pretrained_checkpoint)
    task_suite_names = sorted({row["task_suite_name"] for row in rows})
    assert len(task_suite_names) == 1, (
        f"이 스크립트는 unnorm_key 해석을 위해 manifest 하나당 task_suite 하나만 지원함: {task_suite_names}"
    )

    cfg = GenerateConfig(
        pretrained_checkpoint=local_checkpoint,
        task_suite_name=task_suite_names[0],
        use_l1_regression=True,
        use_proprio=False,  # 합성 이미지에는 실제 proprio가 없음 — 토큰 자체를 뺌 (모듈 docstring 참고)
        use_film=False,
        num_images_in_input=1,  # LIVModule 단일 이미지 전제 (docs/MODEL.md §3.3.1)
        center_crop=True,
    )

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    check_unnorm_key(cfg, model)

    cache_manifest_path = output_dir / "cache_manifest.jsonl"
    n_written, n_failed = 0, 0

    with open(cache_manifest_path, "w") as cache_manifest_f:
        for row in tqdm(rows, desc="caching"):
            image_path = dataset_dir / row["image_path"]
            try:
                image = np.array(PILImage.open(image_path).convert("RGB"))
                sub, vision_features = extract_sample(
                    model, processor, cfg, action_head, image, row["task_description"], args.n_extract_layers
                )
            except Exception as e:  # noqa: BLE001 — 샘플 하나 실패가 전체 캐싱을 막지 않도록
                print(f"[skip] {row['group_id']}/{row['role']}: {e}")
                n_failed += 1
                continue

            cache_fname = f"{row['group_id']}__{row['role']}.npz"
            np.savez_compressed(
                output_dir / cache_fname,
                submatrix=sub,
                vision_features=vision_features,
                object_pos=np.array(row["object_pos"], dtype=np.float32),
                object_quat=np.array(row["object_quat"], dtype=np.float32),
            )

            cache_row = dict(row)
            cache_row["cache_path"] = cache_fname
            cache_row["n_extract_layers_cached"] = args.n_extract_layers
            cache_manifest_f.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nDone. {n_written} cached, {n_failed} failed. cache_manifest: {cache_manifest_path}")


if __name__ == "__main__":
    main()
