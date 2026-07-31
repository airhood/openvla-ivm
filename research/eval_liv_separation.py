"""
research/eval_liv_separation.py

eval_liv.py가 확인한 건 "LIV 임베딩에서 위치를 얼마나 정밀하게 복원할 수 있는가"(L2b probe)였다.
이 스크립트는 다른 질문을 본다: "LIV 임베딩 공간 자체가, IVM이 실제로 쓸 방식(§4.2 metric 기반
검증기: cos(ℓ, e) > threshold)대로 물체 상태가 같은지/다른지를 잘 구분하는가?"

구체적으로:
1. anchor-positive(같은 물체 상태) 쌍의 cosine similarity 분포
   vs anchor-hard_negative(물체 교란됨, translate/rotate/remove별로) 쌍의 분포를 비교.
   두 분포가 잘 갈라질수록(AUC가 1에 가까울수록) threshold 하나로 valid/invalid를 잘 나눌 수 있다는 뜻.
2. hard negative의 실제 물리적 변위량(GT position/quaternion 차이)과 cosine similarity의 상관관계.
   "많이 움직인 물체일수록 임베딩도 많이 달라지는가"를 직접 확인 — L2(contrastive)가 원하는
   바로 그 성질이라, L2b probe보다 이 architecture의 실제 목적에 더 직결된 지표.

사용:
    python research/eval_liv_separation.py --checkpoint research/train_liv_out/liv_checkpoint.pt
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prismatic.models.liv import LIVModule  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from train_liv import LIVCacheDataset, LLM_HIDDEN_DIM, N_HEADS, N_VISION_TOKENS  # noqa: E402


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def auc_separation(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUC: P(랜덤 positive 점수 > 랜덤 negative 점수).

    1.0 = 완벽 분리, 0.5 = 구분 못함(랜덤). sklearn 없이 rank-sum으로 직접 계산.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    combined = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(combined)
    pos_rank_sum = ranks[: len(pos_scores)].sum()
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def quaternion_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1n = q1 / np.linalg.norm(q1)
    q2n = q2 / np.linalg.norm(q2)
    dot = np.clip(abs(np.dot(q1n, q2n)), -1.0, 1.0)
    return float(np.degrees(2 * np.arccos(dot)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt["args"]
    device = torch.device(args.device)

    dataset = LIVCacheDataset(
        train_args["cache_manifest"], max_hard_negatives=train_args["max_hard_negatives"], seed=train_args["seed"]
    )

    liv_module = LIVModule(
        n_heads=N_HEADS,
        n_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
        n_vision_tokens=N_VISION_TOKENS,
        llm_hidden_dim=LLM_HIDDEN_DIM,
        n_extract_layers=train_args["n_extract_layers"],
        pool_size=train_args["pool_size"],
        liv_dim=train_args["liv_dim"],
    ).to(device)
    liv_module.load_state_dict(ckpt["liv_module"])
    liv_module.eval()

    def embed(row):
        d = np.load(row["_cache_dir"] / row["cache_path"])
        sub = torch.from_numpy(d["submatrix"].astype(np.float32)).unsqueeze(0).to(device)
        vis = torch.from_numpy(d["vision_features"].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            return liv_module.forward_from_submatrix(sub, vis).squeeze(0).cpu().numpy()

    pos_sims = []
    neg_sims_by_kind = {"translate": [], "rotate": [], "remove": []}
    displacement_vs_sim = []  # (physical displacement, cosine similarity) — 전체 hard negative 대상

    for sample in dataset.samples:
        anchor_emb = embed(sample["anchor"])
        positive_emb = embed(sample["positive"])
        pos_sims.append(float(np.dot(anchor_emb, positive_emb)))

        anchor_pos = np.array(sample["anchor"]["object_pos"])
        anchor_quat = np.array(sample["anchor"]["object_quat"])

        for hn in sample["hard_negatives"]:
            kind = hn["role"].replace("hard_negative_", "")
            hn_emb = embed(hn)
            sim = float(np.dot(anchor_emb, hn_emb))
            if kind in neg_sims_by_kind:
                neg_sims_by_kind[kind].append(sim)

            pos_disp = float(np.linalg.norm(np.array(hn["object_pos"]) - anchor_pos))
            angle_disp = quaternion_angle_deg(np.array(hn["object_quat"]), anchor_quat)
            # 위치 변위(m)와 각도 변위(deg/180, 0~1 스케일)를 단순 합산해 하나의 "물리적 변위" 스칼라로
            combined_disp = pos_disp + angle_disp / 180.0
            displacement_vs_sim.append((combined_disp, sim))

    pos_sims = np.array(pos_sims)
    all_neg_sims = np.concatenate([np.array(v) for v in neg_sims_by_kind.values() if v])

    print(f"positive pairs: n={len(pos_sims)}, similarity mean={pos_sims.mean():.4f} std={pos_sims.std():.4f}")
    results = {"positive": {"n": len(pos_sims), "mean": float(pos_sims.mean()), "std": float(pos_sims.std())}}

    for kind, sims in neg_sims_by_kind.items():
        if not sims:
            print(f"[{kind}] n=0 (샘플 없음)")
            continue
        sims = np.array(sims)
        auc = auc_separation(pos_sims, sims)
        print(
            f"[{kind}] n={len(sims)}, similarity mean={sims.mean():.4f} std={sims.std():.4f} "
            f"| positive vs {kind} AUC={auc:.3f} (1.0=완벽분리, 0.5=구분안됨)"
        )
        results[kind] = {"n": len(sims), "mean": float(sims.mean()), "std": float(sims.std()), "auc_vs_positive": auc}

    overall_auc = auc_separation(pos_sims, all_neg_sims)
    print(f"\n[전체] positive vs all-hard-negative AUC={overall_auc:.3f}")
    results["overall_auc_positive_vs_hard_negative"] = overall_auc

    disp = np.array([d for d, s in displacement_vs_sim])
    sim = np.array([s for d, s in displacement_vs_sim])
    corr, pval = spearmanr(disp, sim)
    print(f"물리적 변위 vs cosine similarity: Spearman corr={corr:.3f} (p={pval:.2e}) — 음수면 '많이 움직일수록 유사도가 낮아짐'(원하는 방향)")
    results["displacement_vs_similarity_spearman"] = {"n": len(disp), "corr": float(corr), "pvalue": float(pval)}

    log_dir = REPO_ROOT / "research/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = log_dir / f"eval_liv_separation_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "git_commit": git_commit_hash(),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_train_args": train_args,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
