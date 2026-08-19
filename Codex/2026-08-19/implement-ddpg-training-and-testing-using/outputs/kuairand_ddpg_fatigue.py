"""Offline fatigue-aware DDPG for KuaiRand-1K.

This script turns the chronological KuaiRand interaction logs into an offline
RL dataset.  The deterministic actor returns a *desired content profile*;
at serving time, rank a slate by its distance from that profile.  The profile
contains video duration, aspect ratio, music type, and tag, rather than a
video ID, so the learned policy can select previously unseen videos.

The implementation is inspired by the replay-buffer/actor-critic layout of
https://github.com/AIDefender/TSCAC, but is intentionally self-contained and
uses only PyTorch, pandas, and NumPy.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOG_COLUMNS = [
    "user_id", "video_id", "time_ms", "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms",
    "duration_ms",
]
USER_NUMERIC = [
    "is_live_streamer", "is_video_author", "follow_user_num", "fans_user_num",
    "friend_user_num", "register_days",
]
ACTION_COLUMNS = ["duration", "aspect", "music_type", "tag"]


@dataclass
class Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    hidden: int = 256
    batch_size: int = 512
    updates: int = 20_000
    bc_weight: float = 2.5
    seed: int = 7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningStandardizer:
    """Train-set-only standardizer; keeps preprocessing reproducible."""

    def fit(self, values: np.ndarray) -> "RunningStandardizer":
        self.mean = values.mean(axis=0).astype(np.float32)
        self.std = values.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DDPG:
    """Offline DDPG with a small behavior-cloning term for action support."""

    def __init__(self, state_dim: int, action_dim: int, cfg: Config, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.actor = Actor(state_dim, action_dim, cfg.hidden).to(device)
        self.critic = Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.actor_target = Actor(state_dim, action_dim, cfg.hidden).to(device)
        self.critic_target = Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        self.actor.eval()
        result = self.actor(torch.as_tensor(states, device=self.device)).cpu().numpy()
        self.actor.train()
        return result

    def update(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Dict[str, float]:
        state, action, reward, next_state, done = batch
        with torch.no_grad():
            target = reward + self.cfg.gamma * (1.0 - done) * self.critic_target(
                next_state, self.actor_target(next_state)
            )
        critic_loss = F.mse_loss(self.critic(state, action), target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        policy_action = self.actor(state)
        bc_loss = F.mse_loss(policy_action, action)
        actor_loss = -self.critic(state, policy_action).mean() + self.cfg.bc_weight * bc_loss
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        with torch.no_grad():
            for current, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.mul_(1 - self.cfg.tau).add_(self.cfg.tau * current)
            for current, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.mul_(1 - self.cfg.tau).add_(self.cfg.tau * current)
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()), "bc_loss": float(bc_loss.item())}

    def save(self, path: Path, metadata: dict) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": asdict(self.cfg), **metadata}, path)


def read_video_actions(data_dir: Path, needed_ids: set[int]) -> pd.DataFrame:
    """Read only video rows used by logs; avoids retaining the 376 MB CSV."""
    file = data_dir / "video_features_basic_1k.csv"
    chunks: List[pd.DataFrame] = []
    usecols = ["video_id", "video_duration", "server_width", "server_height", "music_type", "tag"]
    for chunk in pd.read_csv(file, usecols=usecols, chunksize=200_000):
        chunk = chunk[chunk.video_id.isin(needed_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    videos = pd.concat(chunks, ignore_index=True).drop_duplicates("video_id")
    videos["duration"] = np.log1p(pd.to_numeric(videos.video_duration, errors="coerce").clip(lower=0))
    width = pd.to_numeric(videos.server_width, errors="coerce")
    height = pd.to_numeric(videos.server_height, errors="coerce")
    videos["aspect"] = width / height.replace(0, np.nan)
    videos["music_type"] = pd.to_numeric(videos.music_type, errors="coerce").fillna(-1)
    # KuaiRand stores multiple tags as strings such as "20,43".  Use the first
    # (primary) tag so every logged action has a fixed, numeric action vector.
    videos["tag"] = pd.to_numeric(
        videos.tag.fillna("").astype(str).str.extract(r"(-?\d+)", expand=False), errors="coerce"
    ).fillna(-1)
    return videos[["video_id", *ACTION_COLUMNS]].replace([np.inf, -np.inf], np.nan).fillna(0)


def static_user_features(data_dir: Path) -> pd.DataFrame:
    users = pd.read_csv(data_dir / "user_features_1k.csv")
    available = [c for c in USER_NUMERIC if c in users]
    result = users[["user_id", *available]].copy()
    for col in available:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
        if col.endswith("_num") or col == "register_days":
            result[col] = np.log1p(result[col].clip(lower=0))
    return result


def fatigue_event(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = frame.duration_ms.to_numpy(np.float32).clip(min=1.0)
    completion = np.clip(frame.play_time_ms.to_numpy(np.float32) / duration, 0, 1)
    engagement = (
        1.0 * frame.is_click + 2.0 * frame.is_like + 2.5 * frame.is_follow +
        1.5 * frame.is_comment + 1.0 * frame.is_forward + 0.5 * frame.long_view + completion
    ).to_numpy(np.float32)
    fatigue = (frame.is_hate.to_numpy(np.float32) + 0.75 * (completion < 0.08).astype(np.float32))
    return engagement, fatigue, completion


def build_transitions(data_dir: Path, log_name: str, max_events: int, history: int, fatigue_penalty: float):
    logs = pd.read_csv(data_dir / log_name, usecols=LOG_COLUMNS, nrows=max_events or None)
    logs = logs.dropna(subset=["user_id", "video_id", "time_ms"]).sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    videos = read_video_actions(data_dir, set(logs.video_id.astype(int)))
    logs = logs.merge(videos, on="video_id", how="inner").merge(static_user_features(data_dir), on="user_id", how="left")
    logs = logs.fillna(0).sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    user_cols = [c for c in USER_NUMERIC if c in logs]
    action_raw = logs[ACTION_COLUMNS].to_numpy(np.float32)
    engagement, event_fatigue, completion = fatigue_event(logs)
    rows: List[dict] = []
    # Per-user chronological split prevents future behavior leaking into training.
    for _, group in logs.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        static = group[user_cols].iloc[0].to_numpy(np.float32)
        prior_actions = [np.zeros(len(ACTION_COLUMNS), np.float32) for _ in range(history)]
        fatigue = 0.0
        for position, i in enumerate(idx):
            state = np.concatenate([static, np.concatenate(prior_actions), [fatigue, completion[i]]]).astype(np.float32)
            reward = engagement[i] - fatigue_penalty * (event_fatigue[i] + fatigue)
            rows.append({"state_raw": state, "action_raw": action_raw[i], "reward": reward,
                         "engagement": engagement[i], "fatigue": event_fatigue[i],
                         "user_id": int(logs.user_id.iloc[i]), "position": position})
            fatigue = max(0.0, 0.85 * fatigue + event_fatigue[i] - 0.15 * float(logs.long_view.iloc[i]))
            prior_actions = (prior_actions + [action_raw[i]])[-history:]
    # Calculate each transition's next state from its following event, terminal at sequence end.
    for a, b in zip(rows, rows[1:]):
        a["next_state_raw"] = b["state_raw"] if a["user_id"] == b["user_id"] else a["state_raw"]
        a["done"] = float(a["user_id"] != b["user_id"])
    rows[-1]["next_state_raw"], rows[-1]["done"] = rows[-1]["state_raw"], 1.0
    # Chronological 70/15/15 train/validation/test split per user.  Validation
    # selects the checkpoint; the untouched final period is used only once.
    counts = pd.Series([r["user_id"] for r in rows]).value_counts().to_dict()
    split = []
    for row in rows:
        n = counts[row["user_id"]]
        if n < 6:
            split.append("train" if row["position"] < n - 1 else "test")
            continue
        train_end = max(1, int(0.70 * n))
        validation_end = min(n - 1, max(train_end + 1, int(0.85 * n)))
        if row["position"] < train_end:
            split.append("train")
        elif row["position"] < validation_end:
            split.append("validation")
        else:
            split.append("test")
    return rows, np.asarray(split)


def arrays_from_rows(rows: Sequence[dict], standardizers: Tuple[RunningStandardizer, RunningStandardizer]):
    state_scaler, action_scaler = standardizers
    state = state_scaler.transform(np.stack([r["state_raw"] for r in rows]))
    next_state = state_scaler.transform(np.stack([r["next_state_raw"] for r in rows]))
    # tanh actor is trained against actions in approximately [-1, 1].
    action = np.clip(action_scaler.transform(np.stack([r["action_raw"] for r in rows])) / 3.0, -1, 1)
    reward = np.asarray([r["reward"] for r in rows], np.float32)[:, None]
    done = np.asarray([r["done"] for r in rows], np.float32)[:, None]
    return state, action, reward, next_state, done


def evaluate(policy: DDPG, test_rows: Sequence[dict], state_scaler: RunningStandardizer, action_scaler: RunningStandardizer, slate_size: int) -> dict:
    """Semi-synthetic OPE: rank observed same-user candidate slates by actor distance."""
    states = state_scaler.transform(np.stack([r["state_raw"] for r in test_rows]))
    desired = policy.act(states)
    candidates = np.clip(action_scaler.transform(np.stack([r["action_raw"] for r in test_rows])) / 3.0, -1, 1)
    action_mse = float(np.mean((desired - candidates) ** 2))
    selected, baseline, slates = [], [], 0
    start = 0
    while start < len(test_rows):
        end = min(start + slate_size, len(test_rows))
        if test_rows[start]["user_id"] != test_rows[end - 1]["user_id"] or end - start < 2:
            start += 1; continue
        distances = ((candidates[start:end] - desired[start]) ** 2).mean(axis=1)
        chosen = start + int(distances.argmin())
        selected.append(test_rows[chosen]); baseline.extend(test_rows[start:end]); slates += 1
        start = end
    def mean(items: Sequence[dict], name: str) -> float:
        return float(np.mean([x[name] for x in items])) if items else float("nan")
    return {
        "evaluation": "observed within-user slate re-ranking (semi-synthetic offline evaluation)",
        "slates": slates, "actor_logged_action_mse": action_mse,
        "policy_selected_reward": mean(selected, "reward"), "policy_selected_engagement": mean(selected, "engagement"),
        "policy_selected_fatigue": mean(selected, "fatigue"), "logged_candidate_reward": mean(baseline, "reward"),
        "logged_candidate_engagement": mean(baseline, "engagement"), "logged_candidate_fatigue": mean(baseline, "fatigue"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate fatigue-aware offline DDPG on KuaiRand-1K")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(r"C:\Users\Internet Cafe\Downloads\KuaiRand-1K\data"),
        help="KuaiRand-1K/data directory (default: supplied local dataset)",
    )
    parser.add_argument("--log", default="log_random_4_22_to_5_08_1k.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/kuairand_ddpg"))
    parser.add_argument("--max-events", type=int, default=0, help="0 reads the complete log; use e.g. 50000 for a quick run")
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--fatigue-penalty", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=2.5)
    parser.add_argument("--slate-size", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=500, help="Validate and consider saving a checkpoint every N updates")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(updates=args.updates, batch_size=args.batch_size, bc_weight=args.bc_weight, seed=args.seed)
    rows, split = build_transitions(args.data_dir, args.log, args.max_events, args.history, args.fatigue_penalty)
    train_rows = [r for r, group in zip(rows, split) if group == "train"]
    validation_rows = [r for r, group in zip(rows, split) if group == "validation"]
    test_rows = [r for r, group in zip(rows, split) if group == "test"]
    if len(train_rows) < cfg.batch_size or not validation_rows or not test_rows:
        raise ValueError("Not enough transitions: increase --max-events or lower --batch-size.")
    state_scaler = RunningStandardizer().fit(np.stack([r["state_raw"] for r in train_rows]))
    action_scaler = RunningStandardizer().fit(np.stack([r["action_raw"] for r in train_rows]))
    train = arrays_from_rows(train_rows, (state_scaler, action_scaler))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DDPG(train[0].shape[1], train[1].shape[1], cfg, device)
    losses = []
    validation_history = []
    best_validation_reward = -float("inf")
    best_step = 0
    best_actor_state = None
    best_critic_state = None
    checkpoint_metadata = {
        "state_mean": state_scaler.mean, "state_std": state_scaler.std,
        "action_mean": action_scaler.mean, "action_std": action_scaler.std,
        "action_columns": ACTION_COLUMNS,
    }
    for step in range(cfg.updates):
        ix = np.random.randint(len(train_rows), size=cfg.batch_size)
        batch = tuple(torch.as_tensor(x[ix], device=device) for x in train)
        result = policy.update(batch)
        current_step = step + 1
        if current_step % 1000 == 0 or step == 0:
            print(f"step {current_step}/{cfg.updates}: {result}")
            losses.append({"step": current_step, **result})
        if current_step % args.eval_every == 0 or current_step == cfg.updates:
            validation = evaluate(policy, validation_rows, state_scaler, action_scaler, args.slate_size)
            validation_history.append({"step": current_step, **validation})
            score = validation["policy_selected_reward"]
            print(f"validation step {current_step}: reward={score:.4f}, fatigue={validation['policy_selected_fatigue']:.4f}")
            if np.isfinite(score) and score > best_validation_reward:
                best_validation_reward, best_step = score, current_step
                best_actor_state = copy.deepcopy(policy.actor.state_dict())
                best_critic_state = copy.deepcopy(policy.critic.state_dict())
                policy.save(args.output_dir / "best_ddpg_fatigue.pt", {
                    **checkpoint_metadata, "best_step": best_step,
                    "selection_metric": "validation_policy_selected_reward",
                    "selection_value": best_validation_reward,
                })
                print(f"saved new best checkpoint at step {best_step}")
    if best_actor_state is None:
        raise RuntimeError("No valid validation slate was available to select a checkpoint. Lower --slate-size.")
    policy.actor.load_state_dict(best_actor_state)
    policy.critic.load_state_dict(best_critic_state)
    metrics = evaluate(policy, test_rows, state_scaler, action_scaler, args.slate_size)
    metrics.update({
        "checkpoint": "best_ddpg_fatigue.pt", "best_step": best_step,
        "best_validation_policy_selected_reward": best_validation_reward,
        "n_transitions": len(rows), "n_train": len(train_rows), "n_validation": len(validation_rows),
        "n_test": len(test_rows), "device": str(device), "config": asdict(cfg),
    })
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output_dir / "losses.json").open("w", encoding="utf-8") as handle:
        json.dump(losses, handle, indent=2)
    with (args.output_dir / "validation_history.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_history, handle, indent=2)
    policy.save(args.output_dir / "ddpg_fatigue_final.pt", checkpoint_metadata)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
