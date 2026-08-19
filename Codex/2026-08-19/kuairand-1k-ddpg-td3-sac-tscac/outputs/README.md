# Fatigue-aware DDPG on KuaiRand-27K

`kuairand_ddpg.py` is a standalone offline DDPG baseline inspired by the data-to-replay-buffer and DDPG evaluation layout of [TSCAC](https://github.com/AIDefender/TSCAC). It intentionally uses current PyTorch APIs and makes the fatigue objective explicit.

The preprocessing step streams the 27K CSV logs, retains a deterministic user subset and most-recent event window per user, then creates chronological `(state, action, reward, next_state, done)` transitions. State includes the last 20 rewards/fatigue/progress values, fatigue average, and time of day. Actions are fixed signed-hash video embeddings, avoiding a 32-million-row embedding table.

The reward is:

`click + 2*like + follow + comment + forward + 0.5*long_view - fatigue_penalty*(short/skip + 0.5*hate)`.

Short/skip is inferred from viewing progress. Adjust `--fatigue-penalty` to set the engagement/fatigue trade-off.

## Install

Use a Python environment with `torch`, `numpy`, and `pandas` installed.

## Run

Download and extract KuaiRand-27K, then run these commands from this directory (replace `python` with the Python executable in your environment if needed):

```powershell
python .\kuairand_ddpg.py prepare --data-dir D:\datasets\KuaiRand-27K --output .\kuairand27k_1k.npz --users 1000 --max-events-per-user 500
python .\kuairand_ddpg.py train --dataset .\kuairand27k_1k.npz --checkpoint .\ddpg_fatigue.pt --steps 100000
python .\kuairand_ddpg.py test --dataset .\kuairand27k_1k.npz --checkpoint .\ddpg_fatigue.pt
```

For a stricter offline action-support guardrail, add `--bc-coef 0.1` to training. With its default of zero this is vanilla DDPG.

`heldout_logged_fatigue` is the actual fatigue rate in held-out logs. `estimated_q_advantage` is only a learned-critic diagnostic, not proof of a real online improvement; offline DDPG can extrapolate beyond logged actions. Validate with randomized-exposure/OPE or an approved online experiment before deployment.

KuaiRand-27K is large (46 GB): the default 1,000 users × 500 most-recent events produces at most 500,000 transitions, making baseline iteration practical.
