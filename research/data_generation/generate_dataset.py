"""
research/data_generation/generate_dataset.py

여러 태스크·초기 상태에 걸쳐 anchor/positive/hard-negative 이미지 세트를 배치 생성한다.
scene_utils.py의 단일-그룹 로직(test_scene_utils.py에서 검증됨)을 여러
(task_suite, task_id, init_state) 조합에 반복 적용해서 이미지 파일 + manifest.jsonl로 저장한다.

로컬 Ubuntu 실행 전용(robosuite 라이브 렌더링 필요, docs/MODEL.md §7 참고). VLA forward pass는
여기서 하지 않는다 — build_liv_cache.py가 별도로 GPU 환경(Colab/A100)에서 이 스크립트의
manifest.jsonl + 이미지를 읽어 수행한다 (1단계=본 스크립트, 2단계=build_liv_cache.py).

대상 물체는 태스크마다 하드코딩하지 않고 env.obj_of_interest(bddl의 :obj_of_interest, LIBERO가
태스크 성공 판정에 쓰는 바로 그 물체 목록)를 그대로 쓴다.

Hard negative는 생성 직후 anchor와의 최소 시각적 차이를 기준으로 걸러낸다(--no_diff_filter로 끔).
회전 대칭 물체(예: 그릇)의 rotate 교란처럼 사실상 anchor와 구별 안 되는 가짜 hard negative가
학습 데이터에 섞이는 것을 방지하기 위함 — docs/MODEL.md §7 "Hard negative의 rotate는 물체 형태에
따라 약한 신호가 될 수 있음" 항목에서 다루기로 한 대응 (1)의 구현.

사용:
    python research/data_generation/generate_dataset.py \
        --task_suite_name libero_spatial \
        --num_tasks 10 \
        --groups_per_task 5 \
        --output_dir research/data_generation/dataset_out
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from scene_utils import (  # noqa: E402
    canonicalize_quaternion_np,
    get_object_pose,
    get_robot_joint_positions,
    object_body_name,
    object_joint_name,
    refresh_observation,
    sample_object_perturbation,
    sample_random_arm_pose,
    set_object_pose,
    set_robot_joint_positions,
)

HARD_NEGATIVE_KINDS = ["translate", "rotate", "remove"]
# test_scene_utils.py에서 실측/보정한 임계값 — 전체 프레임 평균 픽셀 diff (0~255 스케일).
MIN_VISUAL_DIFF = 0.3


def image_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return float(np.abs(img_a.astype(np.int32) - img_b.astype(np.int32)).mean())


def generate_group(env, obj_name: str, group_id: str, rng: np.random.Generator, output_dir: Path, diff_filter: bool):
    """물체 하나 기준 anchor+positive+hard-negative 세트를 생성해서 저장.

    Returns:
        list[dict] — 저장에 성공한 샘플들의 manifest row (task 메타데이터는 호출부에서 채움)
    """
    body = object_body_name(obj_name)
    joint = object_joint_name(obj_name)

    obs = refresh_observation(env)
    anchor_pos, anchor_quat = get_object_pose(env, body)
    anchor_quat = canonicalize_quaternion_np(anchor_quat)
    anchor_arm = get_robot_joint_positions(env)
    img_anchor = get_libero_image(obs)

    group_dir = output_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img_anchor).save(group_dir / "anchor.png")

    rows = [
        {
            "group_id": group_id,
            "role": "anchor",
            "object_name": obj_name,
            "image_path": str((group_dir / "anchor.png").relative_to(output_dir)),
            "object_pos": anchor_pos.tolist(),
            "object_quat": anchor_quat.tolist(),
        }
    ]

    # === positive: 물체 고정, 팔 자세만 랜덤화 ===
    new_arm = sample_random_arm_pose(env, scale=0.3, rng=rng)
    set_robot_joint_positions(env, new_arm)
    obs_pos = refresh_observation(env)
    img_positive = get_libero_image(obs_pos)
    PILImage.fromarray(img_positive).save(group_dir / "positive.png")
    rows.append(
        {
            "group_id": group_id,
            "role": "positive",
            "object_name": obj_name,
            "image_path": str((group_dir / "positive.png").relative_to(output_dir)),
            "object_pos": anchor_pos.tolist(),
            "object_quat": anchor_quat.tolist(),
        }
    )
    set_robot_joint_positions(env, anchor_arm)  # 팔을 anchor 상태로 복원

    # === hard negatives: 팔 고정(anchor와 동일), 물체만 교란 ===
    for kind in HARD_NEGATIVE_KINDS:
        new_pos, new_quat = sample_object_perturbation(anchor_pos, anchor_quat, kind, rng=rng)
        set_object_pose(env, joint, new_pos, new_quat)
        obs_neg = refresh_observation(env)
        img_neg = get_libero_image(obs_neg)
        diff = image_diff(img_neg, img_anchor)

        if diff_filter and diff <= MIN_VISUAL_DIFF:
            # anchor와 사실상 구별 안 되는 가짜 hard negative(주로 회전 대칭 물체의 rotate) -> 스킵.
            set_object_pose(env, joint, anchor_pos, anchor_quat)
            continue

        fname = f"hard_negative_{kind}.png"
        PILImage.fromarray(img_neg).save(group_dir / fname)
        rows.append(
            {
                "group_id": group_id,
                "role": f"hard_negative_{kind}",
                "object_name": obj_name,
                "image_path": str((group_dir / fname).relative_to(output_dir)),
                "object_pos": new_pos.tolist(),
                "object_quat": canonicalize_quaternion_np(new_quat).tolist(),
                "visual_diff_from_anchor": diff,
            }
        )
        set_object_pose(env, joint, anchor_pos, anchor_quat)  # 다음 kind를 위해 복구

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=None, help="처리할 태스크 수 (기본: 전체)")
    parser.add_argument("--groups_per_task", type=int, default=5, help="태스크당 사용할 초기 상태 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "research/data_generation/dataset_out"))
    parser.add_argument(
        "--no_diff_filter",
        action="store_true",
        help="hard negative 최소 시각적 차이 필터를 끈다 (기본: 켜짐, MIN_VISUAL_DIFF 기준)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)

    rng = np.random.default_rng(args.seed)
    total_rows = 0
    skipped_groups = 0

    with open(manifest_path, "w") as manifest_f:
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=256)
            initial_states = task_suite.get_task_init_states(task_id)
            n_states = min(args.groups_per_task, len(initial_states))

            print(f"[task {task_id}/{n_tasks - 1}] {task_description} — {n_states} init state(s)")

            for state_idx in range(n_states):
                env.reset()
                env.set_init_state(initial_states[state_idx])
                obj_names = list(env.obj_of_interest)
                if not obj_names:
                    skipped_groups += 1
                    continue

                for obj_name in obj_names:
                    group_id = f"{args.task_suite_name}_t{task_id}_s{state_idx}_{obj_name}"
                    try:
                        rows = generate_group(
                            env, obj_name, group_id, rng, output_dir, diff_filter=not args.no_diff_filter
                        )
                    except Exception as e:  # noqa: BLE001 — 물체별 실패가 전체 배치를 막지 않도록
                        print(f"  [skip] {group_id}: {e}")
                        skipped_groups += 1
                        continue

                    for row in rows:
                        row["task_suite_name"] = args.task_suite_name
                        row["task_id"] = task_id
                        row["task_description"] = task_description
                        manifest_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_rows += 1

            env.close()

    print(f"\nDone. {total_rows} rows written to {manifest_path} ({skipped_groups} group(s) skipped)")


if __name__ == "__main__":
    main()
