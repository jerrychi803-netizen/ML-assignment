# Fatigue-aware Offline DDPG for KuaiRand-1K

`tscac_ddpg_kuairand.py` is the primary training and test pipeline. It implements the deterministic two-stage constrained actor-critic design in the supplied TSCAC paper: separate critics for WatchTime, Click, Like, Comment, and Hate; auxiliary policies trained in stage one; and a WatchTime policy softly constrained by them in stage two. `kuairand_ddpg_fatigue.py` is retained as the original single-objective baseline.

## What it learns

The deterministic DDPG actor maps a user state to a desired four-dimensional content profile: log video duration, aspect ratio, music type, and tag. A serving system ranks a candidate slate by distance to this desired profile. The state combines static user features, recent content profiles, estimated session fatigue, and last completion. This makes the output usable for recommendations rather than just a score on an item ID.

The reward is:

`engagement - fatigue_penalty * (hate_or_very_short_view + accumulated_session_fatigue)`

where engagement includes click, like, follow, comment, forward, long view, and completion. The actor has a behavior-cloning regularizer, which is important for stable offline DDPG because logged actions define the trusted support.

## Install

Use Python 3.10+ in an environment with PyTorch, NumPy, and pandas (the Codex
built-in Python used to create this deliverable does not bundle PyTorch):

```powershell
pip install torch pandas numpy
```

## Quick smoke run

The random log is appropriate for a first run and small enough to iterate quickly:

```powershell
python .\tscac_ddpg_kuairand.py `
  --data-dir "C:\Users\Internet Cafe\Downloads\KuaiRand-1K\data" `
  --max-events 50000 --stage1-updates 500 --stage2-updates 1500 --batch-size 256 `
  --output-dir .\runs\quick
```

## Full training

```powershell
python .\tscac_ddpg_kuairand.py `
  --data-dir "C:\Users\Internet Cafe\Downloads\KuaiRand-1K\data" `
  --log log_random_4_22_to_5_08_1k.csv `
  --history 20 --stage1-updates 1000 --stage2-updates 3000 --batch-size 512 `
  --bc-weight 5 --constraint-weight 0.1 `
  --output-dir .\runs\tscac_ddpg
```

The script uses a chronological 70/15/15 train/validation/test split. It first trains a supervised behavior-cloning (BC) policy on the training split, then uses BC to rank the held-out candidate slates. Every 500 stage-two updates it selects the model with the best constrained validation score: high WatchTime while keeping Click, Like, and Comment above 95% of the BC policy values and Hate no higher than the BC policy value. It then evaluates that selected checkpoint once on the untouched test period.

Outputs include `best_tscac_ddpg.pt`, `metrics.json`, and `validation_history.json`. Use `best_tscac_ddpg.pt`.

## Evaluation interpretation

The held-out evaluation uses chronological per-user splits and observed, same-user candidate slates. The actor chooses the nearest observed candidate action in each slate, and reports its observed engagement/fatigue beside the average logged candidates. This is a semi-synthetic offline evaluation, not a causal online-lift estimate; validate with an A/B test before deployment.

Key metrics in `metrics.json`:

- `policy_selected_fatigue`: lower is better.
- `policy_selected_engagement`: higher is better.
- `policy_selected_reward`: the fatigue-penalized goal; higher is better.
- `logged_candidate_*`: within-slate logged comparison baseline.

For a more production-oriented system, feed the actor a live candidate embedding containing richer visual/text/audio features, then rank those candidates by actor distance and apply diversity and safety filters.
