"""
research/data_generation/smoke_test.py

scene_utils.py를 짜기 전에, robosuite/LIBERO의 실제 이름/인덱스 규칙을 확인하는 스크립트.
로컬(Windows)엔 robosuite가 없어 여기서 검증할 수 없었던 API 가정들을 여기서 먼저 확인한다:

1. 물체 body/joint 이름이 정확히 뭔지 (env.sim.model.body_names, joint_names)
2. 로봇 팔 관절 인덱스(_ref_joint_pos_indexes)가 실제로 존재하고 맞는지
3. qpos를 직접 바꾸고 env.sim.forward()만 호출해도 렌더링에 반영되는지
   (물리 step 없이 순간이동시키는 방식이 실제로 되는지)

이 스크립트가 통과해야 scene_utils.py의 함수들을 신뢰하고 쓸 수 있다.

중요 — 라이브 렌더링 첫 실제 검증: Phase 0는 사전 렌더링된 pkl로 전환하면서 라이브
렌더링(OffScreenRenderEnv 실제 렌더 호출) 경로를 검증 없이 우회했다. `MUJOCO_GL=egl`은
이 경로에서 두 번 세그폴트가 났었고(Python 예외 없이 프로세스가 죽음), `xvfb+glfw` 조합은
그때 노트북 셀 편집 실수로 실제로는 검증되지 못한 채 넘어갔다. 데이터 생성은 새 이미지를
실시간으로 렌더링해야 해서 pkl로 우회할 수 없으므로, **이 스크립트가 라이브 렌더링을 사실상
처음 제대로 검증하는 자리**다. 반드시 아래처럼 xvfb+glfw로 실행할 것 (plain `python`으로
직접 실행하면 egl 기본 경로를 타서 예전처럼 조용히 죽을 수 있음):

사용 (Colab, xvfb 설치 후):
    !MUJOCO_GL=glfw xvfb-run -a --server-args="-screen 0 1024x768x24" \
        python research/data_generation/smoke_test.py \
        --task_suite_name libero_spatial --task_id 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./research/data_generation/smoke_out")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)

    initial_states = task_suite.get_task_init_states(args.task_id)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_idx])

    print(f"Task: {task_description}")

    # === 1. body/joint 이름 나열 ===
    print("\n=== body_names ===")
    for name in env.sim.model.body_names:
        print(" ", name)

    print("\n=== joint_names ===")
    for name in env.sim.model.joint_names:
        print(" ", name)

    # === 2. 로봇 팔 관절 인덱스 확인 ===
    print("\n=== robot joint indices ===")
    robot = env.robots[0]
    try:
        idx = robot._ref_joint_pos_indexes
        print(f"  robots[0]._ref_joint_pos_indexes = {idx}")
        current_qpos = env.sim.data.qpos[idx].copy()
        print(f"  current joint qpos = {current_qpos}")
    except AttributeError as e:
        print(f"  FAIL: _ref_joint_pos_indexes 없음 ({e}) — robosuite 버전에 따라 속성명이 다를 수 있음. "
              f"robot.__dict__ 또는 dir(robot)에서 'joint' 포함된 속성 찾아볼 것")
        print(f"  robot 객체의 joint 관련 속성: {[a for a in dir(robot) if 'joint' in a.lower()]}")
        return

    # === 3. 렌더링 베이스라인 저장 ===
    img_before = get_libero_image(obs)
    from PIL import Image as PILImage
    PILImage.fromarray(img_before).save(output_dir / "01_before.png")
    print(f"\n저장: {output_dir / '01_before.png'} (변경 전)")

    # === 4. 로봇 팔 관절각을 크게 바꿔보고 렌더링이 실제로 바뀌는지 확인 ===
    new_qpos = current_qpos + 0.5  # 라디안, 눈에 띄게 큰 변화
    env.sim.data.qpos[idx] = new_qpos
    env.sim.forward()  # 물리 step 없이 상태만 반영

    obs_after = env._get_observations()
    img_after = get_libero_image(obs_after)
    PILImage.fromarray(img_after).save(output_dir / "02_after_arm_change.png")
    print(f"저장: {output_dir / '02_after_arm_change.png'} (팔 관절각 변경 후)")

    diff = np.abs(img_after.astype(np.int32) - img_before.astype(np.int32)).mean()
    print(f"\n이미지 평균 차이: {diff:.2f} (0에 가까우면 변경이 렌더링에 반영 안 된 것 — 문제)")
    assert diff > 1.0, "FAIL: 관절각을 바꿨는데 렌더링이 거의 안 바뀜. env.sim.forward()로 충분한지 재확인 필요"

    # 원상복구
    env.sim.data.qpos[idx] = current_qpos
    env.sim.forward()

    # === 5. 물체 pose 읽기/쓰기 시도 (body_names에서 물체로 보이는 것 아무거나) ===
    # LIBERO 물체 이름은 태스크마다 다르므로, body_names 출력에서 로봇/테이블이 아닌 것을 수동으로 골라서
    # 아래 OBJECT_NAME을 채운 뒤 다시 실행해야 함 (1차 실행에서는 이 부분 건너뜀).
    print(
        "\n=== 다음 단계 ===\n"
        "위 body_names 출력에서 조작 대상 물체로 보이는 이름을 찾아서,\n"
        "이 스크립트의 OBJECT_NAME 변수를 채우고 물체 pose get/set도 검증해야 함.\n"
        "(1차 실행 목적: 팔 관절 인덱스 확인 + qpos 직접 조작이 렌더링에 반영되는지 확인)"
    )

    print("\nPASS (팔 관절 부분) — 위 이미지 2장을 비교해서 실제로 팔이 움직였는지 육안으로도 확인할 것")


if __name__ == "__main__":
    main()
