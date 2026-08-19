"""Offline DDPG baseline for fatigue-aware short-video recommendation on KuaiRand-27K.

The program has three subcommands:
  prepare  stream raw KuaiRand CSV files into compact chronological transitions;
  train    train DDPG only from those logged transitions;
  test     report held-out critic and fatigue diagnostics.

This is an offline-RL baseline.  It does not claim that an offline metric is an
online causal effect; deploy a policy only after an online/simulator evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


LOG_COLUMNS = ["user_id", "video_id", "time_ms", "hourmin", "is_click", "is_like",
               "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
               "play_time_ms", "duration_ms"]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def action_features(video_ids: np.ndarray, dim: int) -> np.ndarray:
    """Deterministic signed hashing: no 32M-item embedding table is needed."""
    ids = video_ids.astype(np.uint64).reshape(-1, 1)
    j = np.arange(dim, dtype=np.uint64).reshape(1, -1) + 1
    x = ids * (j * np.uint64(0x9E3779B185EBCA87))
    x ^= x >> np.uint64(33); x *= np.uint64(0xFF51AFD7ED558CCD); x ^= x >> np.uint64(33)
    return ((x & np.uint64(1)).astype(np.float32) * 2.0 - 1.0) / math.sqrt(dim)


def feedback(row: pd.Series, fatigue_penalty: float) -> tuple[float, float]:
    duration = max(float(row.duration_ms), 1.0)
    progress = min(float(row.play_time_ms) / duration, 1.0)
    skip = float(progress < 0.15 or (not bool(row.is_click) and progress < 0.35))
    fatigue = skip + 0.5 * float(row.is_hate)
    engagement = (float(row.is_click) + 2 * float(row.is_like) + float(row.is_follow)
                  + float(row.is_comment) + float(row.is_forward) + 0.5 * float(row.long_view))
    return engagement - fatigue_penalty * fatigue, fatigue


def make_state(history: deque, hourmin: float, history_len: int) -> np.ndarray:
    """State = recent reward/fatigue/progress triples + fatigue EWMA + time of day."""
    pad = [(0.0, 0.0, 0.0)] * (history_len - len(history))
    flat = np.asarray(pad + list(history), dtype=np.float32).reshape(-1)
    fatigue_ewma = 0.0 if not history else float(np.mean([v[1] for v in history]))
    mins = (hourmin // 100) * 60 + (hourmin % 100)
    phase = 2 * math.pi * mins / 1440.0
    return np.concatenate((flat, [fatigue_ewma, math.sin(phase), math.cos(phase)])).astype(np.float32)


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.data_dir); logs = sorted((root / "data").glob(args.log_glob))
    if not logs: raise FileNotFoundError(f"No logs matched {root / 'data' / args.log_glob}")
    user_file = root / "data" / "user_features_27k.csv"
    users = pd.read_csv(user_file, usecols=["user_id"]).user_id.drop_duplicates().sort_values().head(args.users)
    selected = set(users.astype(np.int64).tolist())
    streams: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.max_events_per_user))
    for file in logs:
        print(f"Reading {file.name}", flush=True)
        for chunk in pd.read_csv(file, usecols=LOG_COLUMNS, chunksize=args.chunk_rows):
            chunk = chunk[chunk.user_id.isin(selected)].dropna(subset=["time_ms", "video_id"])
            for user, group in chunk.groupby("user_id", sort=False):
                streams[int(user)].extend(group.to_dict("records"))
    states: list[np.ndarray] = []; actions: list[np.ndarray] = []; rewards: list[float] = []
    next_states: list[np.ndarray] = []; dones: list[float] = []; fatigues: list[float] = []
    for _, events in streams.items():
        events = sorted(events, key=lambda x: x["time_ms"])
        hist: deque = deque(maxlen=args.history_len)
        for i, row in enumerate(events):
            state = make_state(hist, float(row.get("hourmin", 0)), args.history_len)
            reward, fatigue = feedback(pd.Series(row), args.fatigue_penalty)
            progress = min(float(row["play_time_ms"]) / max(float(row["duration_ms"]), 1), 1.0)
            hist_after = deque(hist, maxlen=args.history_len); hist_after.append((reward, fatigue, progress))
            terminal = i == len(events) - 1
            nxt_hour = float(events[i + 1].get("hourmin", 0)) if not terminal else float(row.get("hourmin", 0))
            states.append(state); actions.append(action_features(np.array([row["video_id"]]), args.action_dim)[0])
            rewards.append(reward); next_states.append(make_state(hist_after, nxt_hour, args.history_len))
            dones.append(float(terminal)); fatigues.append(fatigue); hist = hist_after
    if not states: raise ValueError("No transitions found; check --data-dir and --users.")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=np.stack(states), action=np.stack(actions), reward=np.asarray(rewards, np.float32),
                        next_state=np.stack(next_states), done=np.asarray(dones, np.float32), fatigue=np.asarray(fatigues, np.float32))
    print(json.dumps({"transitions": len(states), "state_dim": len(states[0]), "action_dim": args.action_dim,
                      "mean_reward": float(np.mean(rewards)), "mean_fatigue": float(np.mean(fatigues))}, indent=2))


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
    def forward(self, state): return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, state, action): return self.net(torch.cat([state, action], dim=-1))


class DDPG:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device; self.gamma = args.gamma; self.tau = args.tau; self.bc_coef = args.bc_coef
        self.actor = Actor(state_dim, action_dim).to(device); self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device); self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()); self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr); self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
    def update(self, batch):
        s, a, r, ns, d = batch
        with torch.no_grad(): target = r + self.gamma * (1 - d) * self.critic_target(ns, self.actor_target(ns))
        critic_loss = F.mse_loss(self.critic(s, a), target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()
        predicted = self.actor(s); actor_loss = -self.critic(s, predicted).mean() + self.bc_coef * F.mse_loss(predicted, a)
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        with torch.no_grad():
            for online, target_net in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, tp in zip(online.parameters(), target_net.parameters()): tp.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return float(actor_loss.item()), float(critic_loss.item())
    def save(self, path, state_dim, action_dim):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "state_dim": state_dim, "action_dim": action_dim}, path)


def split(data, test_fraction, seed):
    n = len(data["reward"]); rng = np.random.default_rng(seed); idx = rng.permutation(n); cut = int(n * (1 - test_fraction))
    return idx[:cut], idx[cut:]


def batch_from(data, indices, device):
    return tuple(torch.as_tensor(data[k][indices], dtype=torch.float32, device=device).reshape(len(indices), -1 if k != "reward" and k != "done" else 1)
                 for k in ("state", "action", "reward", "next_state", "done"))


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed); data = np.load(args.dataset); train_idx, test_idx = split(data, args.test_fraction, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    agent = DDPG(data["state"].shape[1], data["action"].shape[1], device, args); rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        ids = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=len(train_idx) < args.batch_size)
        aloss, closs = agent.update(batch_from(data, ids, device))
        if step == 1 or step % args.log_every == 0: print(f"step={step} actor_loss={aloss:.4f} critic_loss={closs:.4f}")
    out = Path(args.checkpoint); out.parent.mkdir(parents=True, exist_ok=True); agent.save(out, data["state"].shape[1], data["action"].shape[1]); print(f"saved {out}")
    evaluate(agent, data, test_idx, device)


def evaluate(agent, data, idx, device):
    agent.actor.eval(); agent.critic.eval()
    with torch.no_grad():
        s, a, r, ns, d = batch_from(data, idx, device); logged_q = agent.critic(s, a).mean().item(); policy_q = agent.critic(s, agent.actor(s)).mean().item()
        bellman_mse = F.mse_loss(agent.critic(s, a), r + agent.gamma * (1-d) * agent.critic_target(ns, agent.actor_target(ns))).item()
    print(json.dumps({"heldout_logged_reward": float(data["reward"][idx].mean()), "heldout_logged_fatigue": float(data["fatigue"][idx].mean()),
                      "heldout_logged_q": logged_q, "heldout_policy_q": policy_q, "estimated_q_advantage": policy_q - logged_q,
                      "heldout_bellman_mse": bellman_mse, "note": "Q metrics are offline model diagnostics, not a causal online estimate."}, indent=2))


def test(args):
    data = np.load(args.dataset); _, test_idx = split(data, args.test_fraction, args.seed); device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True); agent = DDPG(ckpt["state_dim"], ckpt["action_dim"], device, args)
    agent.actor.load_state_dict(ckpt["actor"]); agent.critic.load_state_dict(ckpt["critic"]); agent.actor_target.load_state_dict(ckpt["actor"]); agent.critic_target.load_state_dict(ckpt["critic"]); evaluate(agent, data, test_idx, device)


def parser():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--data-dir", required=True, help="Directory containing KuaiRand-27K/data"); q.add_argument("--output", default="kuairand27k_transitions.npz"); q.add_argument("--log-glob", default="log_standard_*27k*.csv"); q.add_argument("--users", type=int, default=1000); q.add_argument("--max-events-per-user", type=int, default=500); q.add_argument("--chunk-rows", type=int, default=250000); q.add_argument("--history-len", type=int, default=20); q.add_argument("--action-dim", type=int, default=32); q.add_argument("--fatigue-penalty", type=float, default=1.0); q.set_defaults(func=prepare)
    for name, fn in (("train", train), ("test", test)):
        q = sub.add_parser(name); q.add_argument("--dataset", required=True); q.add_argument("--checkpoint", required=True); q.add_argument("--seed", type=int, default=7); q.add_argument("--test-fraction", type=float, default=.2); q.add_argument("--cpu", action="store_true"); q.add_argument("--gamma", type=float, default=.99); q.add_argument("--tau", type=float, default=.005); q.add_argument("--actor-lr", type=float, default=1e-4); q.add_argument("--critic-lr", type=float, default=1e-3); q.add_argument("--bc-coef", type=float, default=0.0, help="Optional behavior-cloning guardrail; 0 is vanilla DDPG."); q.set_defaults(func=fn)
        if name == "train": q.add_argument("--steps", type=int, default=100000); q.add_argument("--batch-size", type=int, default=512); q.add_argument("--log-every", type=int, default=1000)
    return p


if __name__ == "__main__":
    args = parser().parse_args(); args.func(args)
