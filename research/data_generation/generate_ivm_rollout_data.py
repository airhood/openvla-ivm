"""
research/data_generation/generate_ivm_rollout_data.py

Phase 2(IVM) 데이터 생성 — VLA를 실제로 굴려서 rollout하고, 청크 경계마다 (학습 완료, frozen된)
LIVModule로 ℓ을 계산하고, 청크 실행 구간에서 valid/invalid 프레임을 뽑는다
(docs/MODEL.md §6 Phase 2 스텝 개요 / "Phase 2 데이터 파이프라인 설계 확정" 참고).

실행 환경: robosuite 라이브 렌더링 + VLA(7B) GPU가 **동시에** 필요하다 — rollout은 "관측 → VLA가
action 결정 → 물리 스텝 → 새 관측"이 매 순간 한 프로세스 안에서 일어나야 하는 닫힌 루프라, Phase 1처럼
렌더링(로컬)과 VLA(Colab) 실행 시점을 분리할 수 없다. Colab은 GPU는 있지만 robosuite EGL 렌더링이
안 되고(vendor ICD 없음), 이 로컬 머신은 렌더링은 되지만 GPU가 4GB뿐이라 7B를 못 올린다 — 실제
디스플레이/드라이버가 있는 GPU 서버가 필요.

**미검증**: 이 스크립트는 그런 서버가 없어서 작성 시점에 한 번도 끝까지 실행해본 적 없다. VLA
forward pass(get_action)와 LIVModule 연결 부분은 verify_liv_wiring.py/build_liv_cache.py에서
이미 검증된 패턴을 그대로 재사용했고, 물리 상태 스냅샷/복구/perturbation 부분은 scene_utils.py의
이미 개별 검증된 함수들을 그대로 쓰지만, 전체 파이프라인을 통째로 실행해본 적은 없다.

설계 요약:
- 청크 경계(action_queue가 빎)마다: 그 시점 관측으로 `get_action(liv_module=...)` 호출 →
  action chunk + ℓ(`vla.last_liv`)을 한 번의 forward로 같이 얻음. ℓ은 청크당 1번만 계산.
- 청크 실행 중(다음 NUM_ACTIONS_CHUNK 스텝): 매 스텝 실제 action을 실행해서 나온 진짜 다음 관측을
  valid로 저장. 그리고 `--invalid_prob` 확률로, **실제 rollout에는 영향을 주지 않는 방식으로**
  (get_sim_state()로 진짜 상태를 스냅샷 → 물체 perturbation 적용 → 렌더링 → 원상복구) invalid도
  같이 생성.
- Invalid perturbation은 gross(remove/replace) + medium/fine(translate/rotate, scene_utils의
  PERTURBATION_TIERS) 중 랜덤 선택 — docs/MODEL.md §7 난이도 커리큘럼 대응.
- use_proprio=False로 강제 — LIVModule이 Phase 1 캐싱 때 이 설정(proprio 토큰 없음)으로 학습됐으므로
  inference 때도 동일 시퀀스 레이아웃을 맞춰야 학습-추론 분포가 어긋나지 않음 (실제 proprio가
  있어도 일부러 안 씀 — 나중에 검증되면 재고 가능).

사용:
    python research/data_generation/generate_ivm_rollout_data.py \
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --liv_checkpoint research/train_liv_out_gross_fine/liv_checkpoint.pt \
        --task_suite_name libero_spatial \
        --num_episodes 20 \
        --output_dir research/data_generation/ivm_rollout_out \
        --load_in_8bit
"""

import argparse
import json
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "research"))

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image  # noqa: E402
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    TASK_MAX_STEPS,
    GenerateConfig,
    check_unnorm_key,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.robot_utils import get_action  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from phase0_attention_heatmap import resolve_local_checkpoint  # noqa: E402
from prismatic.models.liv import LIVModule  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from scene_utils import (  # noqa: E402
    apply_object_replacement,
    canonicalize_quaternion_np,
    get_object_pose,
    object_body_name,
    object_joint_name,
    refresh_observation,
    sample_object_perturbation,
    set_object_pose,
)

LLM_HIDDEN_DIM = 4096
N_HEADS = 32
N_VISION_TOKENS = 256


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_frozen_liv_module(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    liv_module = LIVModule(
        n_heads=N_HEADS,
        n_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
        n_vision_tokens=N_VISION_TOKENS,
        llm_hidden_dim=LLM_HIDDEN_DIM,
        n_extract_layers=train_args["n_extract_layers"],
        pool_size=train_args["pool_size"],
        liv_dim=train_args["liv_dim"],
    ).to(device, dtype=torch.bfloat16)
    liv_module.load_state_dict(ckpt["liv_module"])
    liv_module.eval()
    for p in liv_module.parameters():
        p.requires_grad_(False)
    return liv_module


def sample_invalid_frame(env, obj_name: str, rng: np.random.Generator):
    """현재 진행 중인 rollout의 물리 상태를 건드리지 않고 invalid 프레임 하나를 만든다.

    Returns:
        (image, kind, tier) 또는 생성 실패 시 None (예: replace인데 씬에 다른 물체가 없음)
    """
    live_state = env.get_sim_state()  # 실제 rollout 상태 스냅샷 — 끝나면 반드시 이 상태로 복귀
    body = object_body_name(obj_name)
    joint = object_joint_name(obj_name)
    cur_pos, cur_quat = get_object_pose(env, body)
    cur_quat = canonicalize_quaternion_np(cur_quat)

    kind = rng.choice(["translate", "rotate", "remove", "replace"])
    tier = None

    if kind == "replace":
        replacement_name, restore_fn = apply_object_replacement(env, obj_name, cur_pos, cur_quat, rng=rng)
        if replacement_name is None:
            env.regenerate_obs_from_state(live_state)
            return None
        obs = refresh_observation(env)
        img = get_libero_image(obs)
        restore_fn()
    else:
        tier = "gross" if kind == "remove" else rng.choice(["gross", "fine"])
        new_pos, new_quat = sample_object_perturbation(cur_pos, cur_quat, kind, tier=tier, rng=rng)
        set_object_pose(env, joint, new_pos, new_quat)
        obs = refresh_observation(env)
        img = get_libero_image(obs)
        set_object_pose(env, joint, cur_pos, cur_quat)

    # perturbation 중 sim이 바뀌었을 수 있으므로, 실제 rollout 상태로 확실하게 복귀
    env.regenerate_obs_from_state(live_state)
    return img, kind, tier


def run_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, liv_module, output_dir, manifest_f, episode_meta, rng, invalid_prob, role_counts):
    env.reset()
    obs = env.set_init_state(episode_meta["initial_state"])

    obj_names = list(env.obj_of_interest)
    if not obj_names:
        return 0
    target_obj = obj_names[0]  # 여러 개면 첫 번째만 사용 (단순화)

    action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
    t = 0
    chunk_idx = -1
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    episode_dir = output_dir / episode_meta["episode_id"]
    episode_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)

            if len(action_queue) == 0:
                chunk_idx += 1
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head, proprio_projector=proprio_projector,
                    liv_module=liv_module,
                )
                action_queue.extend(actions)

                liv_vec = model.last_liv.squeeze(0).float().cpu().numpy().tolist()
                anchor_fname = f"chunk{chunk_idx:03d}_anchor.png"
                PILImage.fromarray(img).save(episode_dir / anchor_fname)
                manifest_f.write(json.dumps({
                    **episode_meta_for_row(episode_meta),
                    "chunk_id": f"{episode_meta['episode_id']}_c{chunk_idx:03d}",
                    "role": "anchor",
                    "image_path": str((episode_dir / anchor_fname).relative_to(output_dir)),
                    "liv": liv_vec,
                }, ensure_ascii=False) + "\n")
                n_rows += 1
                role_counts["anchor"] += 1

            action = action_queue.popleft()
            step_in_chunk = NUM_ACTIONS_CHUNK - len(action_queue)  # 1..NUM_ACTIONS_CHUNK
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())

            valid_img = get_libero_image(obs)
            valid_fname = f"chunk{chunk_idx:03d}_step{step_in_chunk:02d}_valid.png"
            PILImage.fromarray(valid_img).save(episode_dir / valid_fname)
            manifest_f.write(json.dumps({
                **episode_meta_for_row(episode_meta),
                "chunk_id": f"{episode_meta['episode_id']}_c{chunk_idx:03d}",
                "role": "valid",
                "image_path": str((episode_dir / valid_fname).relative_to(output_dir)),
                "step_in_chunk": step_in_chunk,
            }, ensure_ascii=False) + "\n")
            n_rows += 1
            role_counts["valid"] += 1

            if rng.random() < invalid_prob:
                result = sample_invalid_frame(env, target_obj, rng)
                if result is not None:
                    inv_img, kind, tier = result
                    inv_fname = f"chunk{chunk_idx:03d}_step{step_in_chunk:02d}_invalid_{kind}.png"
                    PILImage.fromarray(inv_img).save(episode_dir / inv_fname)
                    manifest_f.write(json.dumps({
                        **episode_meta_for_row(episode_meta),
                        "chunk_id": f"{episode_meta['episode_id']}_c{chunk_idx:03d}",
                        "role": "invalid",
                        "image_path": str((episode_dir / inv_fname).relative_to(output_dir)),
                        "step_in_chunk": step_in_chunk,
                        "perturbation_kind": kind,
                        "perturbation_tier": tier,
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
                    role_counts[f"invalid_{kind}"] += 1

            if done:
                break
            t += 1
    except Exception as e:  # noqa: BLE001 — 에피소드 하나 실패가 전체를 막지 않도록
        print(f"  [episode error] {episode_meta['episode_id']}: {e}")

    return n_rows


def episode_meta_for_row(episode_meta: dict) -> dict:
    return {k: v for k, v in episode_meta.items() if k != "initial_state"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--liv_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--episodes_per_task", type=int, default=5)
    parser.add_argument("--invalid_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "research/data_generation/ivm_rollout_out"))
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    local_checkpoint = resolve_local_checkpoint(args.pretrained_checkpoint)
    cfg = GenerateConfig(
        pretrained_checkpoint=local_checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=True,
        use_proprio=False,  # LIVModule이 Phase 1에서 이 설정으로 학습됨 - inference도 맞춰야 함
        use_film=False,
        num_images_in_input=1,  # LIVModule 단일 이미지 전제 (docs/MODEL.md §3.3.1)
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"Loading VLA from {cfg.pretrained_checkpoint} ...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    check_unnorm_key(cfg, model)
    device = next(model.parameters()).device

    print(f"Loading frozen LIVModule from {args.liv_checkpoint} ...")
    liv_module = load_frozen_liv_module(args.liv_checkpoint, device)

    from experiments.robot.robot_utils import get_image_resize_size
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)

    rng = np.random.default_rng(args.seed)
    total_rows = 0
    role_counts = {"anchor": 0, "valid": 0}
    for kind in ["translate", "rotate", "remove", "replace"]:
        role_counts[f"invalid_{kind}"] = 0

    with open(manifest_path, "w") as manifest_f:
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)
            n_episodes = min(args.episodes_per_task, len(initial_states))
            print(f"[task {task_id}/{n_tasks - 1}] {task_description} — {n_episodes} episode(s)")

            for ep_idx in range(n_episodes):
                episode_meta = {
                    "episode_id": f"{args.task_suite_name}_t{task_id}_ep{ep_idx}",
                    "task_suite_name": args.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "initial_state": initial_states[ep_idx],
                }
                n_rows = run_episode(
                    cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                    liv_module, output_dir, manifest_f, episode_meta, rng, args.invalid_prob, role_counts,
                )
                total_rows += n_rows
                manifest_f.flush()

            env.close()

    print(f"\nDone. {total_rows} rows written to {manifest_path}")
    print(f"role counts: {role_counts}")

    log_dir = REPO_ROOT / "research/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = log_dir / f"generate_ivm_rollout_data_{timestamp}_{args.task_suite_name}.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "git_commit": git_commit_hash(),
            "args": vars(args),
            "manifest_path": str(manifest_path.resolve()),
            "total_rows": total_rows,
            "role_counts": role_counts,
        }, f, indent=2, ensure_ascii=False)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
