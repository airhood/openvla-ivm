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
3. 변위량을 구간(bin)으로 나눠서, 구간별로 positive 분포 대비 AUC + Wasserstein distance를 본다.
   변위가 0에 가까운 hard negative는 사실상 positive와 구분이 안 되는 게 정상(같은 상태나 다름
   없으므로) — 그 "구분이 무너지기 시작하는 변위량"이 곧 지금 모델의 실질적 민감도(해상도)다.
   AUC가 구간별로 얼마나 완만하게/급격하게 0.5(랜덤)로 떨어지는지가 "divergence 감소율".

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
from scipy.stats import rankdata, spearmanr, wasserstein_distance

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
    parser.add_argument(
        "--cache_manifest",
        type=str,
        nargs="+",
        default=None,
        help="평가에 쓸 캐시 manifest(들). 생략하면 체크포인트가 학습에 썼던 manifest 그대로 사용. "
        "학습 때 안 본 데이터(예: fine-tier perturbation)로 일반화 여부를 볼 때 지정",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt["args"]
    device = torch.device(args.device)

    eval_manifest = args.cache_manifest if args.cache_manifest is not None else train_args["cache_manifest"]
    if args.cache_manifest is not None:
        print(f"체크포인트의 학습 manifest 대신 지정된 manifest로 평가: {eval_manifest}")

    dataset = LIVCacheDataset(
        eval_manifest, max_hard_negatives=train_args["max_hard_negatives"], seed=train_args["seed"]
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
    # kind별 "자기 고유 단위"의 변위 — translate는 위치(m)만, rotate는 각도(deg)만 움직이므로
    # combined_disp처럼 섞지 않고 각자의 진짜 물리량으로 따로 봄(2026-08-01, kind 혼합이 구간별
    # 분석을 왜곡한다는 문제 발견 후 추가)
    native_disp_by_kind = {"translate": [], "rotate": []}  # (displacement, similarity)

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

            if kind == "translate":
                native_disp_by_kind["translate"].append((pos_disp, sim))
            elif kind == "rotate":
                native_disp_by_kind["rotate"].append((angle_disp, sim))

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

    def binned_separation(disp: np.ndarray, sim: np.ndarray, n_bins: int, label: str):
        """변위량을 n_bins개 분위 구간으로 나눠 구간별 positive 대비 AUC/Wasserstein을 계산+출력."""
        quantile_edges = np.quantile(disp, np.linspace(0, 1, n_bins + 1))
        quantile_edges = np.unique(quantile_edges)  # 값이 중복되면(예: remove가 다 -1.0 근처) bin이 줄어듦
        bin_idx = np.digitize(disp, quantile_edges[1:-1], right=True)

        print(f"\n[{label}] 변위량 구간별 positive 대비 분리도 (작을수록 AUC가 0.5에 가까워야 정상):")
        bin_results = []
        for b in range(len(quantile_edges) - 1):
            mask = bin_idx == b
            if mask.sum() < 3:
                continue
            bin_sims = sim[mask]
            bin_disp = disp[mask]
            auc = auc_separation(pos_sims, bin_sims)
            wdist = float(wasserstein_distance(pos_sims, bin_sims))
            print(
                f"  변위 [{bin_disp.min():.4f}, {bin_disp.max():.4f}] (n={mask.sum():3d}): "
                f"similarity mean={bin_sims.mean():.4f} | AUC={auc:.3f} | Wasserstein={wdist:.4f}"
            )
            bin_results.append(
                {
                    "displacement_min": float(bin_disp.min()),
                    "displacement_max": float(bin_disp.max()),
                    "displacement_mean": float(bin_disp.mean()),
                    "n": int(mask.sum()),
                    "similarity_mean": float(bin_sims.mean()),
                    "auc_vs_positive": auc,
                    "wasserstein_distance_vs_positive": wdist,
                }
            )
        return bin_results

    # === 변위량 구간별 AUC/divergence — "구분이 무너지는 지점"(민감도 해상도) 확인 ===
    # (1) 전체 pooled(translate+rotate 혼합, combined_disp) — 기존 방식, 비교용으로 유지
    results["displacement_bins"] = binned_separation(disp, sim, n_bins=8, label="전체(translate+rotate 혼합)")

    # (2) kind별 — translate는 위치(m), rotate는 각도(deg) 고유 단위로 따로 봄. combined_disp로
    # 섞으면 대칭 물체의 rotate(실질적 무변화)가 translate와 같은 구간에 섞여 분석이 왜곡됨
    # (2026-08-01, gross 학습 체크포인트를 fine 데이터로 평가했을 때 중간 구간 AUC가 0.5 밑으로
    # 떨어지는 현상 발견 후 원인 확인용으로 추가)
    for kind, pairs in native_disp_by_kind.items():
        if len(pairs) < 10:
            continue
        kind_disp = np.array([d for d, s in pairs])
        kind_sim = np.array([s for d, s in pairs])
        unit = "m" if kind == "translate" else "deg"
        results[f"displacement_bins_{kind}"] = binned_separation(
            kind_disp, kind_sim, n_bins=6, label=f"{kind} 단독 ({unit})"
        )

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
                "eval_manifest": [str(Path(p).resolve()) for p in eval_manifest],
                "eval_manifest_overridden": args.cache_manifest is not None,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
