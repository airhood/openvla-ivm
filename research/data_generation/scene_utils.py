"""
research/data_generation/scene_utils.py

robosuite/LIBERO 씬 상태를 직접 조작해서 positive/hard-negative 이미지 쌍과
물체 GT를 생성하는 유틸리티. Phase 1(LIV)과 Phase 2(IVM) 데이터 생성이 이 모듈을
공유한다 — 둘 다 "씬 상태 설정 → 렌더링 → GT 추출"이라는 같은 하부 동작이 필요하기 때문.

실행 환경: 로컬 Ubuntu(실제 디스플레이 세션). Colab에서는 robosuite의 EGL offscreen
렌더링이 막혀서(NVIDIA EGL vendor ICD 없음) 여기서 쓸 수 없다 — docs/MODEL.md §7 참고.

API는 2026-07-30에 실제 LIBERO 환경(libero_spatial task 0)에서 직접 검증됨:
- 물체 body 이름: "{object_name}_main" (예: "akita_black_bowl_1_main")
- 물체 free-joint 이름: "{object_name}_joint0" (예: "akita_black_bowl_1_joint0")
- quaternion 순서: (w, x, y, z) — MuJoCo 관례, canonicalize_quaternion()과 동일 규약
"""

import numpy as np


def get_object_pose(env, object_name: str):
    """물체의 현재 world pose(GT)를 읽는다.

    Args:
        env: robosuite/LIBERO 환경 (OffScreenRenderEnv)
        object_name: 물체 body 이름, "_main" 접미사 포함 (예: "akita_black_bowl_1_main")

    Returns:
        position: (3,) ndarray
        quaternion: (4,) ndarray, (w, x, y, z) 순서
    """
    position = env.sim.data.get_body_xpos(object_name).copy()
    quaternion = env.sim.data.get_body_xquat(object_name).copy()
    return position, quaternion


def set_object_pose(env, object_joint_name: str, position: np.ndarray, quaternion: np.ndarray) -> None:
    """물체를 특정 pose로 순간이동시킨다 (자유 관절 직접 설정, 물리 step 없이).

    Args:
        env: robosuite/LIBERO 환경
        object_joint_name: 물체의 free joint 이름, "_joint0" 접미사 포함
                            (예: "akita_black_bowl_1_joint0")
        position: (3,) — world 좌표
        quaternion: (4,) — (w, x, y, z)
    """
    qpos = np.concatenate([np.asarray(position), np.asarray(quaternion)])
    env.sim.data.set_joint_qpos(object_joint_name, qpos)
    env.sim.forward()  # 물리 step 없이 상태만 반영 (렌더링에 즉시 적용되도록)


def object_body_name(object_name: str) -> str:
    """물체 기본 이름 -> body 이름 (예: "akita_black_bowl_1" -> "akita_black_bowl_1_main")."""
    return f"{object_name}_main"


def object_joint_name(object_name: str) -> str:
    """물체 기본 이름 -> free joint 이름 (예: "akita_black_bowl_1" -> "akita_black_bowl_1_joint0")."""
    return f"{object_name}_joint0"


def get_robot_joint_positions(env) -> np.ndarray:
    """로봇 팔의 현재 관절각을 읽는다."""
    robot = env.robots[0]
    return env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()


def set_robot_joint_positions(env, joint_positions: np.ndarray) -> None:
    """로봇 팔 관절각을 직접 설정한다 (IK나 정책 없이, 순간이동)."""
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = joint_positions
    env.sim.forward()


def sample_random_arm_pose(env, scale: float = 0.3, rng: np.random.Generator = None) -> np.ndarray:
    """현재 관절각 기준으로 랜덤 오프셋을 준 자세를 샘플링한다.

    Args:
        env: robosuite/LIBERO 환경
        scale: 관절각 랜덤 오프셋의 표준편차 (radian)
        rng: numpy Generator (재현성용, 없으면 새로 생성)

    Returns:
        (n_joints,) — 새 관절각
    """
    rng = rng or np.random.default_rng()
    current = get_robot_joint_positions(env)
    offset = rng.normal(scale=scale, size=current.shape)
    return current + offset


def refresh_observation(env):
    """qpos를 직접 수정한 뒤 새 관측값(렌더링 포함)을 받는다.

    OffScreenRenderEnv(ControlEnv 래퍼)는 _get_observations()를 직접 노출하지 않는다.
    regenerate_obs_from_state()가 set_state -> sim.forward() -> observables 갱신 ->
    _get_observations()를 캡슐화한 공식 경로라 이걸 통해 받는다. 이미 env.sim.data.qpos를
    직접 수정해둔 상태이므로 get_sim_state()로 그 상태를 그대로 스냅샷해서 넘긴다.
    """
    return env.regenerate_obs_from_state(env.get_sim_state())


def canonicalize_quaternion_np(quat: np.ndarray) -> np.ndarray:
    """quaternion을 w>=0 반구로 정규화한다 (q와 -q의 double-cover 문제 해소).

    prismatic.models.liv.canonicalize_quaternion의 numpy/데이터생성 쪽 대응.
    L2b GT를 저장하기 전에 반드시 이걸 거쳐야 한다.
    """
    quat = np.asarray(quat)
    sign = -1.0 if quat[0] < 0 else 1.0
    return quat * sign


def sample_object_perturbation(
    position: np.ndarray,
    quaternion: np.ndarray,
    kind: str,
    rng: np.random.Generator = None,
):
    """hard negative용 물체 교란. kind에 따라 이동/회전/제거를 적용한다.

    Args:
        position: (3,) 원래 위치
        quaternion: (4,) 원래 자세 (w,x,y,z)
        kind: "translate" | "rotate" | "remove"
        rng: numpy Generator

    Returns:
        new_position: (3,)
        new_quaternion: (4,)
    """
    rng = rng or np.random.default_rng()
    position = np.asarray(position, dtype=float).copy()
    quaternion = np.asarray(quaternion, dtype=float).copy()

    if kind == "translate":
        # 테이블 평면(x,y) 위에서 확실히 눈에 띄는 정도로 이동. z(높이)는 유지.
        offset = rng.uniform(low=-0.15, high=0.15, size=2)
        # 원점 근처로 이동하는 것을 피하기 위해 최소 이동량 보장
        offset = np.sign(offset) * np.clip(np.abs(offset), 0.08, None)
        position[0] += offset[0]
        position[1] += offset[1]
        return position, quaternion

    if kind == "rotate":
        # z축 기준 큰 회전 (예: 90도 근방)
        angle = rng.choice([-1, 1]) * (np.pi / 2 + rng.uniform(-0.2, 0.2))
        half = angle / 2
        rot_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # z축 회전, (w,x,y,z)
        new_quat = quaternion_multiply(rot_quat, quaternion)
        return position, new_quat

    if kind == "remove":
        # 카메라 시야 밖 멀리(테이블 아래)로 순간이동 -> "물체가 사라진" 상태
        position = position.copy()
        position[2] -= 1.0
        return position, quaternion

    raise ValueError(f"unknown perturbation kind: {kind}")


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """quaternion 곱셈, (w,x,y,z) 순서. q1을 q2에 적용(q1 이후 q2 순서 아님 — q1 * q2)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
