"""
research/eval_liv.py

train_liv.py로 학습한 체크포인트의 L2b(물체 상태 예측) 오차를 물리 단위(미터, 도)로 확인한다.
loss 곡선(MSE)만으로는 "실제로 몇 cm/몇 도 틀리는지" 감이 안 오기 때문에, Phase 2(IVM)로
넘어가기 전에 LIV가 인코딩한 물체 상태 정보가 실제로 쓸만한 정밀도인지 점검하는 용도.

체크포인트에 저장된 args(cache_manifest, seed, val_frac 등)를 그대로 읽어서 train_liv.py와
동일한 train/val split을 재현한다 — LIVCacheDataset 구성이 결정적(construction 순서가 파일
읽는 순서 그대로)이라 같은 seed면 같은 split이 나옴.

사용:
    python research/eval_liv.py --checkpoint research/train_liv_out/liv_checkpoint.pt
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prismatic.models.liv import LIVModule, ObjectStateDecoder  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from train_liv import LIVCacheDataset, LLM_HIDDEN_DIM, N_HEADS, N_VISION_TOKENS  # noqa: E402


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def quaternion_angle_error_deg(pred_quat: np.ndarray, gt_quat: np.ndarray) -> np.ndarray:
    """두 quaternion(w,x,y,z) 사이의 회전각 오차(도). pred는 unit quaternion이 아닐 수 있어 정규화.

    Args:
        pred_quat, gt_quat: (N, 4)
    Returns:
        (N,) — 도 단위 각도 오차, [0, 180]
    """
    pred_norm = pred_quat / np.linalg.norm(pred_quat, axis=-1, keepdims=True).clip(min=1e-8)
    dot = np.abs(np.sum(pred_norm * gt_quat, axis=-1)).clip(-1.0, 1.0)  # |q1 . q2| — double-cover 무시
    return np.degrees(2 * np.arccos(dot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt["args"]
    device = torch.device(args.device)

    print(f"Checkpoint args: {json.dumps(train_args, ensure_ascii=False)}")

    dataset = LIVCacheDataset(
        train_args["cache_manifest"], max_hard_negatives=train_args["max_hard_negatives"], seed=train_args["seed"]
    )
    n_val = max(1, int(len(dataset) * train_args["val_frac"])) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(train_args["seed"])
    )
    print(f"dataset: {len(dataset)} groups -> train {n_train} / val {n_val} (재현된 split)")

    liv_module = LIVModule(
        n_heads=N_HEADS,
        n_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
        n_vision_tokens=N_VISION_TOKENS,
        llm_hidden_dim=LLM_HIDDEN_DIM,
        n_extract_layers=train_args["n_extract_layers"],
        pool_size=train_args["pool_size"],
        liv_dim=train_args["liv_dim"],
    ).to(device)
    decoder = ObjectStateDecoder(liv_dim=train_args["liv_dim"], hidden_dim=256, state_dim=7).to(device)
    liv_module.load_state_dict(ckpt["liv_module"])
    decoder.load_state_dict(ckpt["decoder"])
    liv_module.eval()
    decoder.eval()

    def evaluate(subset, name):
        pos_errors_m, angle_errors_deg = [], []
        with torch.no_grad():
            for i in range(len(subset)):
                sample = subset[i]
                anchor_sub = torch.from_numpy(sample["anchor_sub"]).unsqueeze(0).to(device)
                anchor_vis = torch.from_numpy(sample["anchor_vis"]).unsqueeze(0).to(device)
                gt_state = sample["anchor_state"]  # (7,) = pos(3) + quat(4)

                liv = liv_module.forward_from_submatrix(anchor_sub, anchor_vis)
                pred_state = decoder(liv).squeeze(0).cpu().numpy()

                pos_err = float(np.linalg.norm(pred_state[:3] - gt_state[:3]))
                angle_err = float(quaternion_angle_error_deg(pred_state[3:][None, :], gt_state[3:][None, :])[0])
                pos_errors_m.append(pos_err)
                angle_errors_deg.append(angle_err)

        pos_errors_m = np.array(pos_errors_m)
        angle_errors_deg = np.array(angle_errors_deg)
        stats = {
            "n_samples": len(subset),
            "position_error_m": {
                "mean": float(pos_errors_m.mean()),
                "median": float(np.median(pos_errors_m)),
                "std": float(pos_errors_m.std()),
                "max": float(pos_errors_m.max()),
            },
            "angle_error_deg": {
                "mean": float(angle_errors_deg.mean()),
                "median": float(np.median(angle_errors_deg)),
                "std": float(angle_errors_deg.std()),
                "max": float(angle_errors_deg.max()),
            },
        }
        print(
            f"[{name}] n={len(subset)} | position error: mean={stats['position_error_m']['mean']*100:.2f}cm "
            f"median={stats['position_error_m']['median']*100:.2f}cm max={stats['position_error_m']['max']*100:.2f}cm "
            f"| angle error: mean={stats['angle_error_deg']['mean']:.1f}deg median={stats['angle_error_deg']['median']:.1f}deg"
        )
        return stats

    train_stats = evaluate(train_set, "train")
    val_stats = evaluate(val_set, "val")

    log_dir = REPO_ROOT / "research/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = log_dir / f"eval_liv_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "git_commit": git_commit_hash(),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_train_args": train_args,
                "train": train_stats,
                "val": val_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
