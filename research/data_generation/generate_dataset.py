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

Domain Randomization(--no_domain_randomization으로 끔, 기본 켜짐, MODEL.md §7 "선택이 아닌 필수"):
robosuite `DomainRandomizationWrapper`로 텍스처/카메라/조명을 랜덤화. **그룹당 1회**만 랜덤화하고
같은 그룹의 anchor/positive/hard-negative는 전부 같은 도메인(배경/조명/카메라)을 공유한다 — 논문
근거의 "frame-wise"(rollout의 매 프레임 독립 랜덤화, 배경 의존을 끊기 위함)를 문자 그대로 그룹
내부까지 적용하면, hard negative 최소 시각적 차이 필터와 L2b GT 비교가 "물체가 실제로 움직였는가"가
아니라 "배경이 우연히 얼마나 바뀌었는가"에 지배당해 필터 자체가 무의미해진다(예: 회전 대칭 물체의
가짜 hard negative를 걸러내는 로직이 깨짐). 그룹(=하나의 씬 비교 단위) 단위로 랜덤화하면 그룹
내부 비교는 여전히 "물체/팔 변화만 다른 변수"로 깨끗하게 유지되면서, 데이터셋 전체로 보면 매
그룹마다 독립적인 배경을 보게 되어 원 취지(배경 의존 학습 방지)는 그대로 달성된다.
동적 물성 랜덤화(질량/마찰 등, `randomize_dynamics`)는 렌더링에 영향 없고 MODEL.md §7이 명시한
범위(조명/텍스처/카메라)도 아니므로 사용하지 않는다. Table distractor 랜덤화는 `DomainRandomizationWrapper`
범위 밖(물체 배치 자체를 바꿔야 함)이라 아직 미구현 — 별도 TODO.

Perturbation tier(--perturbation_tier, 기본 gross): 2026-07-31 추가. `eval_liv_separation.py`로
확인해보니 기존(gross) hard negative는 전부 변위가 커서(translate≥8cm, rotate≈90도) LIV 임베딩이
"상태가 바뀌었는지"를 구분하는 능력이 어느 변위부터 무너지는지 관측을 못 함 — 가장 작은 변위
구간에서도 이미 AUC 0.99+였음. fine tier(translate 1~5cm, rotate ~15도)로 그 경계를 찾기 위한
옵션. **fine 사용 시 --no_diff_filter를 같이 켜는 걸 권장** — MIN_VISUAL_DIFF 필터가 gross 기준
(회전 대칭 물체의 가짜 negative를 거르기 위함)이라 fine의 진짜-작지만-실재하는 변화까지 같이
걸러버림. gross와 fine을 나중에 합쳐 쓸 걸 대비해 group_id에 tier suffix를 붙임(gross는 하위호환을
위해 접미어 없음, fine만 "_fine").

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
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from robosuite.wrappers import DomainRandomizationWrapper  # noqa: E402
from scene_utils import (  # noqa: E402
    canonicalize_quaternion_np,
    enable_shadows,
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


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def image_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return float(np.abs(img_a.astype(np.int32) - img_b.astype(np.int32)).mean())


def wrap_domain_randomization(env, seed: int):
    """텍스처/카메라/조명만 랜덤화(모듈 docstring 참고 — 동적 물성/table distractor는 범위 밖).

    주의 — 이 wrapper의 `.reset()`은 절대 호출하지 말 것: robosuite는 `env.reset()` 시
    `hard_reset=True`(기본값)라서 매번 완전히 새 `MjSim`을 만들고 옛 sim은 `.free()`로 속성까지
    지워버리는데(`MjSim.free()`가 `del self.model`), `DomainRandomizationWrapper.reset()`은
    `save_default_domain()`(옛 sim 참조 사용)을 `update_sim()`(새 sim으로 갱신)보다 먼저 호출해서
    `AttributeError: 'MjSim' object has no attribute 'model'`이 남. 그래서 reset/set_init_state는
    항상 raw env에서 직접 호출하고, 이 wrapper는 `.randomize_domain()` 전용으로만 쓴다 — 그때마다
    `refresh_modders_sim()`으로 현재 sim을 먼저 넣어줘야 한다(매 env.reset()마다 sim이 바뀌므로).
    """
    return DomainRandomizationWrapper(
        env,
        seed=seed,
        randomize_color=True,
        randomize_camera=True,
        randomize_lighting=True,
        randomize_dynamics=False,
        randomize_on_reset=False,
        randomize_every_n_steps=0,
    )


def refresh_modders_sim(dr_wrapper, env) -> None:
    """env.reset()으로 sim이 교체된 뒤, wrapper 내부 modder들이 새 sim을 보게 갱신."""
    for modder in dr_wrapper.modders:
        modder.update_sim(env.sim)


def generate_group(
    env,
    obj_name: str,
    group_id: str,
    rng: np.random.Generator,
    output_dir: Path,
    diff_filter: bool,
    dr_wrapper=None,
    perturbation_tier: str = "gross",
):
    """물체 하나 기준 anchor+positive+hard-negative 세트를 생성해서 저장.

    Returns:
        list[dict] — 저장에 성공한 샘플들의 manifest row (task 메타데이터는 호출부에서 채움)
    """
    body = object_body_name(obj_name)
    joint = object_joint_name(obj_name)

    if dr_wrapper is not None:
        # 그룹당 1회만 랜덤화 — 이유는 모듈 docstring "Domain Randomization" 항목 참고
        # (그룹 내부는 같은 배경 공유해야 diff 필터/L2b GT 비교가 물체 변화만 반영함)
        refresh_modders_sim(dr_wrapper, env)
        dr_wrapper.randomize_domain()

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
        new_pos, new_quat = sample_object_perturbation(anchor_pos, anchor_quat, kind, tier=perturbation_tier, rng=rng)
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
                "perturbation_tier": perturbation_tier,
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
    parser.add_argument(
        "--no_domain_randomization",
        action="store_true",
        help="텍스처/카메라/조명 domain randomization을 끈다 (기본: 켜짐, MODEL.md §7 참고)",
    )
    parser.add_argument(
        "--perturbation_tier",
        type=str,
        default="gross",
        choices=["gross", "fine"],
        help="hard negative 교란 강도 (scene_utils.PERTURBATION_TIERS 참고). "
        "gross=기존 기본값(큰 이동/90도 회전), fine=작은 이동/미세 회전 — MODEL.md §7 난이도 커리큘럼 대응. "
        "remove는 tier와 무관. fine으로 생성할 땐 diff_filter가 작은 변화를 지워버리므로 --no_diff_filter도 같이 켜는 걸 권장",
    )
    args = parser.parse_args()
    dr_enabled = not args.no_domain_randomization

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)

    rng = np.random.default_rng(args.seed)
    total_rows = 0
    skipped_groups = 0
    role_counts = Counter()

    with open(manifest_path, "w") as manifest_f:
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=256)
            dr_wrapper = wrap_domain_randomization(env, seed=args.seed + task_id) if dr_enabled else None
            initial_states = task_suite.get_task_init_states(task_id)
            n_states = min(args.groups_per_task, len(initial_states))

            print(f"[task {task_id}/{n_tasks - 1}] {task_description} — {n_states} init state(s)")

            for state_idx in range(n_states):
                env.reset()
                enable_shadows(env)  # env.reset()이 매번 새 모델을 만들어 castshadow가 초기화됨
                env.set_init_state(initial_states[state_idx])
                obj_names = list(env.obj_of_interest)
                if not obj_names:
                    skipped_groups += 1
                    continue

                for obj_name in obj_names:
                    # gross(기본값)는 기존 group_id 그대로 유지(이미 생성된 캐시와의 호환성).
                    # fine은 접미어를 붙여 별도 그룹으로 구분 — 같은 (task,state,obj)라도 gross/fine을
                    # 나중에 한 LIVCacheDataset에 합칠 때 group_id 충돌로 서로 덮어쓰지 않게 하기 위함.
                    tier_suffix = "" if args.perturbation_tier == "gross" else f"_{args.perturbation_tier}"
                    group_id = f"{args.task_suite_name}_t{task_id}_s{state_idx}_{obj_name}{tier_suffix}"
                    try:
                        rows = generate_group(
                            env,
                            obj_name,
                            group_id,
                            rng,
                            output_dir,
                            diff_filter=not args.no_diff_filter,
                            dr_wrapper=dr_wrapper,
                            perturbation_tier=args.perturbation_tier,
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
                        role_counts[row["role"]] += 1

            env.close()

    print(f"\nDone. {total_rows} rows written to {manifest_path} ({skipped_groups} group(s) skipped)")
    print(f"role counts: {dict(role_counts)}")

    log_dir = REPO_ROOT / "research/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = log_dir / f"generate_dataset_{timestamp}_{args.task_suite_name}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "git_commit": git_commit_hash(),
                "args": vars(args),
                "manifest_path": str(manifest_path.resolve()),
                "total_rows": total_rows,
                "role_counts": dict(role_counts),
                "skipped_groups": skipped_groups,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
