"""Paper-aligned deterministic TSCAC-DDPG for KuaiRand-1K.

Based on Cai et al., "Two-Stage Constrained Actor-Critic for Short Video
Recommendation" (WWW 2023).  This implementation uses one critic/policy for
each auxiliary response in stage 1, then trains a WatchTime policy in stage 2
that is softly constrained to remain close to those auxiliary policies.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from kuairand_ddpg_fatigue import ACTION_COLUMNS, LOG_COLUMNS, USER_NUMERIC, Actor, Critic, RunningStandardizer, read_video_actions, static_user_features

RESPONSE_NAMES = ("watch_time", "click", "like", "comment", "anti_hate")
AUXILIARIES = RESPONSE_NAMES[1:]


@dataclass
class Config:
    gamma_main: float = .99
    gamma_aux: float = .95
    tau: float = .005
    actor_lr: float = 5e-5
    critic_lr: float = 2e-4
    hidden: int = 256
    batch_size: int = 512
    bc_weight: float = 5.0
    constraint_weight: float = .10
    seed: int = 7


class Agent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.cfg, self.device = cfg, device
        self.actor, self.critic = Actor(state_dim, action_dim, cfg.hidden).to(device), Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.actor_target, self.critic_target = Actor(state_dim, action_dim, cfg.hidden).to(device), Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()); self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, state):
        return self.actor(torch.as_tensor(state, device=self.device)).cpu().numpy()

    def critic_step(self, s, a, r, ns, d, gamma):
        with torch.no_grad(): target = r + gamma * (1 - d) * self.critic_target(ns, self.actor_target(ns))
        loss = F.mse_loss(self.critic(s, a), target)
        self.critic_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10); self.critic_opt.step()
        return float(loss.item())

    def soft_update(self):
        with torch.no_grad():
            for src, dst in zip(self.actor.parameters(), self.actor_target.parameters()): dst.mul_(1-self.cfg.tau).add_(self.cfg.tau*src)
            for src, dst in zip(self.critic.parameters(), self.critic_target.parameters()): dst.mul_(1-self.cfg.tau).add_(self.cfg.tau*src)

    def snapshot(self):
        return {name: copy.deepcopy(getattr(self, name).state_dict()) for name in ("actor", "critic", "actor_target", "critic_target")}
    def restore(self, saved):
        for name in saved: getattr(self, name).load_state_dict(saved[name])


def build_rows(data_dir: Path, log_name: str, max_events: int, history: int):
    logs = pd.read_csv(data_dir / log_name, usecols=LOG_COLUMNS, nrows=max_events or None).dropna(subset=["user_id", "video_id", "time_ms"])
    logs = logs.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    logs = logs.merge(read_video_actions(data_dir, set(logs.video_id.astype(int))), on="video_id", how="inner").merge(static_user_features(data_dir), on="user_id", how="left").fillna(0)
    logs = logs.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    user_columns = [c for c in USER_NUMERIC if c in logs]; action = logs[ACTION_COLUMNS].to_numpy(np.float32)
    completion = np.clip(logs.play_time_ms.to_numpy(np.float32) / logs.duration_ms.to_numpy(np.float32).clip(min=1), 0, 1)
    # Main signal WatchTime; sparse auxiliaries follow the paper. Anti-hate is
    # maximized, equivalently constraining the actual Hate response downward.
    response = np.stack((completion, logs.is_click, logs.is_like, logs.is_comment, -logs.is_hate), 1).astype(np.float32)
    rows = []
    for _, group in logs.groupby("user_id", sort=False):
        index = group.index.to_numpy(); static = group[user_columns].iloc[0].to_numpy(np.float32)
        actions = [np.zeros(len(ACTION_COLUMNS), np.float32) for _ in range(history)]; feedback = [0., 0., 0.]
        for pos, i in enumerate(index):
            state = np.concatenate((static, np.concatenate(actions), feedback)).astype(np.float32)
            rows.append({"state": state, "action": action[i], "response": response[i], "user": int(logs.user_id.iloc[i]), "pos": pos})
            actions = (actions + [action[i]])[-history:]; feedback = [completion[i], float(logs.is_like.iloc[i]), float(logs.is_hate.iloc[i])]
    for x, y in zip(rows, rows[1:]): x["next_state"], x["done"] = (y["state"], 0.) if x["user"] == y["user"] else (x["state"], 1.)
    rows[-1]["next_state"], rows[-1]["done"] = rows[-1]["state"], 1.
    counts = pd.Series([r["user"] for r in rows]).value_counts().to_dict(); groups = []
    for r in rows:
        n, p = counts[r["user"]], r["pos"]
        if n < 6: groups.append("train" if p < n-1 else "test"); continue
        a, b = max(1, int(.70*n)), min(n-1, max(int(.70*n)+1, int(.85*n)))
        groups.append("train" if p < a else "validation" if p < b else "test")
    return rows, np.asarray(groups)


def make_arrays(rows, state_scaler, action_scaler, response_scaler):
    s = state_scaler.transform(np.stack([r["state"] for r in rows])); ns = state_scaler.transform(np.stack([r["next_state"] for r in rows]))
    a = np.clip(action_scaler.transform(np.stack([r["action"] for r in rows])) / 3, -1, 1)
    r = response_scaler.transform(np.stack([x["response"] for x in rows])); d = np.asarray([x["done"] for x in rows], np.float32)[:, None]
    return s, a, r, ns, d


def evaluate(agent, behavior_actor, rows, state_scaler, action_scaler, slate_size, device):
    states = state_scaler.transform(np.stack([r["state"] for r in rows])); preference = agent.act(states)
    with torch.no_grad(): behavior_preference = behavior_actor(torch.as_tensor(states, device=device)).cpu().numpy()
    candidates = np.clip(action_scaler.transform(np.stack([r["action"] for r in rows])) / 3, -1, 1)
    picked, baseline, slates, start = [], [], 0, 0
    while start < len(rows):
        end = min(start+slate_size, len(rows))
        if end-start < 2 or rows[start]["user"] != rows[end-1]["user"]: start += 1; continue
        chosen = start + int((candidates[start:end] @ preference[start]).argmax())
        bc_chosen = start + int((candidates[start:end] @ behavior_preference[start]).argmax())
        picked.append(rows[chosen]); baseline.append(rows[bc_chosen]); slates += 1; start = end
    def aggregate(items, prefix):
        values = np.stack([x["response"] for x in items]) if items else np.full((1, 5), np.nan)
        result = {f"{prefix}_{name}": float(values[:, i].mean()) for i, name in enumerate(RESPONSE_NAMES)}
        result[f"{prefix}_hate"] = -result.pop(f"{prefix}_anti_hate"); return result
    out = {"evaluation": "same-user observed candidate-slate ranking against behavior cloning (offline proxy)", "slates": slates, "actor_logged_action_mse": float(((preference-candidates)**2).mean())}
    out.update(aggregate(picked, "policy")); out.update(aggregate(baseline, "bc")); return out


def score(metrics, aux_ratio, hate_ratio):
    penalty = 0.
    for name in ("click", "like", "comment"):
        base = metrics[f"bc_{name}"]
        if base > 1e-6: penalty += max(0., aux_ratio*base-metrics[f"policy_{name}"])/base
    base = metrics["bc_hate"]
    if base > 1e-6: penalty += max(0., metrics["policy_hate"]-hate_ratio*base)/base
    return metrics["policy_watch_time"] - penalty


def main():
    p = argparse.ArgumentParser(description="Paper-aligned two-stage constrained DDPG for KuaiRand-1K")
    p.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\Internet Cafe\Downloads\KuaiRand-1K\data")); p.add_argument("--log", default="log_random_4_22_to_5_08_1k.csv")
    p.add_argument("--output-dir", type=Path, default=Path("runs/tscac_ddpg")); p.add_argument("--max-events", type=int, default=0); p.add_argument("--history", type=int, default=20)
    p.add_argument("--bc-updates", type=int, default=1000); p.add_argument("--stage1-updates", type=int, default=1000); p.add_argument("--stage2-updates", type=int, default=3000); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--bc-weight", type=float, default=5.); p.add_argument("--constraint-weight", type=float, default=.10); p.add_argument("--eval-every", type=int, default=500); p.add_argument("--slate-size", type=int, default=5)
    p.add_argument("--aux-ratio", type=float, default=.95); p.add_argument("--hate-ratio", type=float, default=1.); p.add_argument("--seed", type=int, default=7); args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(batch_size=args.batch_size, bc_weight=args.bc_weight, constraint_weight=args.constraint_weight, seed=args.seed)
    rows, split = build_rows(args.data_dir, args.log, args.max_events, args.history)
    train_rows = [x for x, g in zip(rows, split) if g == "train"]; val_rows = [x for x, g in zip(rows, split) if g == "validation"]; test_rows = [x for x, g in zip(rows, split) if g == "test"]
    if len(train_rows) < args.batch_size or not val_rows or not test_rows: raise ValueError("Need more data or a smaller batch size.")
    ss = RunningStandardizer().fit(np.stack([x["state"] for x in train_rows])); ac = RunningStandardizer().fit(np.stack([x["action"] for x in train_rows])); rs = RunningStandardizer().fit(np.stack([x["response"] for x in train_rows]))
    arrays = make_arrays(train_rows, ss, ac, rs); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Supervised behavior-cloning baseline (paper Sec. 5.1): predict the
    # continuous logged preference action, then rank candidate videos by dot product.
    behavior_actor = Actor(arrays[0].shape[1], arrays[1].shape[1], cfg.hidden).to(device)
    behavior_opt = torch.optim.Adam(behavior_actor.parameters(), lr=cfg.actor_lr)
    for step in range(args.bc_updates):
        index = np.random.randint(len(train_rows), size=args.batch_size); state = torch.as_tensor(arrays[0][index], device=device); action = torch.as_tensor(arrays[1][index], device=device)
        behavior_loss = F.mse_loss(behavior_actor(state), action)
        behavior_opt.zero_grad(); behavior_loss.backward(); torch.nn.utils.clip_grad_norm_(behavior_actor.parameters(), 10); behavior_opt.step()
        if step == 0 or (step + 1) % 500 == 0: print(f"bc {step+1}/{args.bc_updates}: action_mse={behavior_loss.item():.4f}")
    behavior_actor.eval()
    agents = {name: Agent(arrays[0].shape[1], arrays[1].shape[1], cfg, device) for name in AUXILIARIES}
    # Stage 1: individual auxiliary policies and critics (TSCAC Sec. 4.2).
    for step in range(args.stage1_updates):
        index = np.random.randint(len(train_rows), size=args.batch_size); s,a,r,ns,d = (torch.as_tensor(x[index], device=device) for x in arrays)
        report = {}
        for response_i, name in enumerate(AUXILIARIES, start=1):
            agent = agents[name]; c = agent.critic_step(s,a,r[:, response_i:response_i+1],ns,d,cfg.gamma_aux); pred = agent.actor(s); bc = F.mse_loss(pred,a); loss = -agent.critic(s,pred).mean()+cfg.bc_weight*bc
            agent.actor_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),10); agent.actor_opt.step(); agent.soft_update(); report[name] = round(c,4)
        if step == 0 or (step+1) % 500 == 0: print(f"stage1 {step+1}/{args.stage1_updates}: critic losses {report}")
    main_agent = Agent(arrays[0].shape[1], arrays[1].shape[1], cfg, device); best, best_step, best_state, history = -float("inf"), 0, None, []
    # Stage 2: WatchTime critic plus Gaussian-kernel-equivalent proximity loss (paper Eq. 10).
    for step in range(args.stage2_updates):
        index = np.random.randint(len(train_rows), size=args.batch_size); s,a,r,ns,d = (torch.as_tensor(x[index], device=device) for x in arrays)
        c = main_agent.critic_step(s,a,r[:,:1],ns,d,cfg.gamma_main); pred = main_agent.actor(s); bc = F.mse_loss(pred,a)
        with torch.no_grad(): targets = [agent.actor(s) for agent in agents.values()]
        proximity = sum(F.mse_loss(pred,t) for t in targets)/len(targets); loss = -main_agent.critic(s,pred).mean()+cfg.bc_weight*bc+cfg.constraint_weight*proximity
        main_agent.actor_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(main_agent.actor.parameters(),10); main_agent.actor_opt.step(); main_agent.soft_update(); current=step+1
        if current == 1 or current % 500 == 0: print(f"stage2 {current}/{args.stage2_updates}: actor={loss.item():.4f}, critic={c:.4f}, bc={bc.item():.4f}, proximity={proximity.item():.4f}")
        if current % args.eval_every == 0 or current == args.stage2_updates:
            v = evaluate(main_agent,behavior_actor,val_rows,ss,ac,args.slate_size,device); candidate = score(v,args.aux_ratio,args.hate_ratio); history.append({"step":current,"constrained_score":candidate,**v}); print(f"validation {current}: score={candidate:.4f}, watch={v['policy_watch_time']:.4f}, hate={v['policy_hate']:.4f}")
            if np.isfinite(candidate) and candidate > best:
                best,best_step,best_state=candidate,current,main_agent.snapshot(); torch.save({"main":best_state,"auxiliary":{k:x.snapshot() for k,x in agents.items()},"behavior_actor":behavior_actor.state_dict(),"config":asdict(cfg),"best_stage2_step":best_step,"validation_constrained_score":best,"state_mean":ss.mean,"state_std":ss.std,"action_mean":ac.mean,"action_std":ac.std,"action_columns":ACTION_COLUMNS},args.output_dir/"best_tscac_ddpg.pt"); print("saved best_tscac_ddpg.pt")
    if best_state is None: raise RuntimeError("No valid validation slate; reduce --slate-size.")
    main_agent.restore(best_state); result=evaluate(main_agent,behavior_actor,test_rows,ss,ac,args.slate_size,device); result.update({"checkpoint":"best_tscac_ddpg.pt","best_stage2_step":best_step,"best_validation_constrained_score":best,"n_train":len(train_rows),"n_validation":len(val_rows),"n_test":len(test_rows),"device":str(device),"config":asdict(cfg)})
    (args.output_dir/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (args.output_dir/"validation_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
