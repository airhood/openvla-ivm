"""
research/train_liv.py

Phase 1 학습 루프: build_liv_cache.py가 만든 캐시(attention 서브행렬 + vision feature,
그리고 물체 GT)만 읽어서 LIVModule + ObjectStateDecoder(L2b)를 학습한다. VLA(7B)는 이미
캐싱 단계에서 한 번만 forward됐으므로 여기서는 전혀 쓰지 않는다 — docs/MODEL.md §6
"Phase 1 스텝 개요"의 1(VLA forward)이 build_liv_cache.py로 이미 끝난 상태에서 이어지는
2~4단계(LIVModule → L2/L2b → backward)만 수행.

LIVModule이 작은 모델(~6.5M 파라미터)이라 GPU 없이 CPU로도, 또는 소형 GPU로도 돌아간다
(VLA 7B와 달리 여기는 A100/Colab이 필요 없음).

Hard negative 개수(K)는 그룹마다 다르다(0~3개, 필터링 때문에 더 적을 수 있음, MODEL.md §7 참고).
LIVContrastiveLoss는 고정 K 텐서를 기대하므로, 그룹마다 있는 hard negative 중에서
--max_hard_negatives개를 복원추출로 채운다(zero-padding 대신 — 가짜 negative를 만들지 않기 위함).
Hard negative가 하나도 없는 그룹은 학습에서 제외한다(현재 데이터에서는 드묾 — translate가 거의
항상 살아남으므로).

사용:
    python research/train_liv.py \
        --cache_manifest research/data_generation/liv_cache_out/cache_manifest.jsonl \
        --output_dir research/train_liv_out \
        --epochs 20
"""

import argparse
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"

from prismatic.models.liv import (  # noqa: E402
    LIVContrastiveLoss,
    LIVModule,
    ObjectStateDecoder,
    ObjectStateLoss,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402

LLM_HIDDEN_DIM = 4096  # Llama-2-7B — 캐싱에 쓴 체크포인트와 일치해야 함
N_HEADS = 32
N_VISION_TOKENS = 256


class LIVCacheDataset(Dataset):
    """cache_manifest.jsonl을 그룹(group_id)별로 묶어서 anchor/positive/hard_negative 세트로 노출."""

    def __init__(self, cache_manifest_path: str, max_hard_negatives: int = 3, seed: int = 0):
        cache_manifest_path = Path(cache_manifest_path)
        self.cache_dir = cache_manifest_path.parent
        self.max_hard_negatives = max_hard_negatives
        self.rng = np.random.default_rng(seed)

        rows = [json.loads(line) for line in cache_manifest_path.read_text().splitlines() if line.strip()]
        groups = defaultdict(dict)
        for row in rows:
            groups[row["group_id"]][row["role"]] = row

        self.samples = []
        for group_id, roles in groups.items():
            if "anchor" not in roles or "positive" not in roles:
                continue
            hn_rows = [v for k, v in roles.items() if k.startswith("hard_negative")]
            if not hn_rows:
                continue
            self.samples.append({"group_id": group_id, "anchor": roles["anchor"], "positive": roles["positive"], "hard_negatives": hn_rows})

    def __len__(self):
        return len(self.samples)

    def _load(self, row):
        d = np.load(self.cache_dir / row["cache_path"])
        return d["submatrix"].astype(np.float32), d["vision_features"].astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        anchor_sub, anchor_vis = self._load(s["anchor"])
        pos_sub, pos_vis = self._load(s["positive"])

        hn_rows = s["hard_negatives"]
        k = self.max_hard_negatives
        pick = self.rng.choice(len(hn_rows), size=k, replace=len(hn_rows) < k)
        hn_sub, hn_vis = [], []
        for i in pick:
            sub, vis = self._load(hn_rows[i])
            hn_sub.append(sub)
            hn_vis.append(vis)

        anchor_state = np.array(
            s["anchor"]["object_pos"] + s["anchor"]["object_quat"], dtype=np.float32
        )  # (7,) = position(3) + canonicalized quaternion(4)

        return {
            "anchor_sub": anchor_sub,
            "anchor_vis": anchor_vis,
            "positive_sub": pos_sub,
            "positive_vis": pos_vis,
            "hn_sub": np.stack(hn_sub),  # (K, L, H, N)
            "hn_vis": np.stack(hn_vis),  # (K, N, D)
            "anchor_state": anchor_state,
        }


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(loader, liv_module, decoder, contrastive_loss, state_loss, optimizer, scheduler, device, lambda_l2b, grad_clip, train: bool):
    liv_module.train(train)
    decoder.train(train)

    total_loss, total_l2, total_l2b, n_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        anchor_sub = batch["anchor_sub"].to(device)
        anchor_vis = batch["anchor_vis"].to(device)
        pos_sub = batch["positive_sub"].to(device)
        pos_vis = batch["positive_vis"].to(device)
        hn_sub = batch["hn_sub"].to(device)  # (B, K, L, H, N)
        hn_vis = batch["hn_vis"].to(device)  # (B, K, N, D)
        anchor_state = batch["anchor_state"].to(device)

        with torch.set_grad_enabled(train):
            liv_anchor = liv_module.forward_from_submatrix(anchor_sub, anchor_vis)
            liv_positive = liv_module.forward_from_submatrix(pos_sub, pos_vis)

            B, K = hn_sub.shape[0], hn_sub.shape[1]
            hn_sub_flat = hn_sub.reshape(B * K, *hn_sub.shape[2:])
            hn_vis_flat = hn_vis.reshape(B * K, *hn_vis.shape[2:])
            liv_hard_neg = liv_module.forward_from_submatrix(hn_sub_flat, hn_vis_flat).reshape(B, K, -1)

            l2 = contrastive_loss(liv_anchor, liv_positive, liv_hard_neg)

            pred_state = decoder(liv_anchor)
            l2b = state_loss(pred_state, anchor_state)

            loss = l2 + lambda_l2b * l2b

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(liv_module.parameters()) + list(decoder.parameters()), max_norm=grad_clip
            )
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        total_l2 += l2.item()
        total_l2b += l2b.item()
        n_batches += 1

    return total_loss / n_batches, total_l2 / n_batches, total_l2b / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "research/train_liv_out"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_l2b", type=float, default=1.0)
    parser.add_argument("--n_extract_layers", type=int, default=4, help="캐시가 담은 레이어 수 이하여야 함 (ablation 후보 1/4/8)")
    parser.add_argument("--pool_size", type=int, default=8)
    parser.add_argument("--liv_dim", type=int, default=128)
    parser.add_argument("--max_hard_negatives", type=int, default=3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="학습 파라미터+epoch별 loss를 남길 JSONL 경로. 기본: "
        "research/logs/train_liv_<timestamp>.jsonl (git 추적 대상 — checkpoint와 달리 로그는 커밋함)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = Path(args.log_file) if args.log_file else REPO_ROOT / "research/logs" / f"train_liv_{timestamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")

    def log(record: dict):
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LIVCacheDataset(args.cache_manifest, max_hard_negatives=args.max_hard_negatives, seed=args.seed)
    n_val = max(1, int(len(dataset) * args.val_frac)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"dataset: {len(dataset)} groups usable (>=1 hard negative) -> train {n_train} / val {n_val}")

    log(
        {
            "type": "run_start",
            "timestamp": timestamp,
            "git_commit": git_commit_hash(),
            "args": vars(args),
            "cache_manifest": str(Path(args.cache_manifest).resolve()),
            "n_groups_total": len(dataset),
            "n_groups_train": n_train,
            "n_groups_val": n_val,
        }
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if n_val > 0 else None

    liv_module = LIVModule(
        n_heads=N_HEADS,
        n_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
        n_vision_tokens=N_VISION_TOKENS,
        llm_hidden_dim=LLM_HIDDEN_DIM,
        n_extract_layers=args.n_extract_layers,
        pool_size=args.pool_size,
        liv_dim=args.liv_dim,
    ).to(device)
    decoder = ObjectStateDecoder(liv_dim=args.liv_dim, hidden_dim=256, state_dim=7).to(device)

    contrastive_loss = LIVContrastiveLoss(temperature=args.temperature)
    state_loss = ObjectStateLoss()

    params = list(liv_module.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_cosine_lr(step, args.warmup_steps, total_steps)
    )

    for epoch in range(args.epochs):
        train_loss, train_l2, train_l2b = run_epoch(
            train_loader, liv_module, decoder, contrastive_loss, state_loss, optimizer, scheduler, device, args.lambda_l2b, args.grad_clip, train=True
        )
        msg = f"[epoch {epoch}] train loss={train_loss:.4f} (L2={train_l2:.4f}, L2b={train_l2b:.4f})"
        record = {"type": "epoch", "epoch": epoch, "train_loss": train_loss, "train_l2": train_l2, "train_l2b": train_l2b}
        if val_loader is not None:
            val_loss, val_l2, val_l2b = run_epoch(
                val_loader, liv_module, decoder, contrastive_loss, state_loss, optimizer, scheduler, device, args.lambda_l2b, args.grad_clip, train=False
            )
            msg += f" | val loss={val_loss:.4f} (L2={val_l2:.4f}, L2b={val_l2b:.4f})"
            record.update({"val_loss": val_loss, "val_l2": val_l2, "val_l2b": val_l2b})
        print(msg)
        log(record)

    ckpt_path = output_dir / "liv_checkpoint.pt"
    torch.save(
        {
            "liv_module": liv_module.state_dict(),
            "decoder": decoder.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint to {ckpt_path}")
    log({"type": "run_end", "checkpoint_path": str(ckpt_path.resolve())})
    log_f.close()
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
