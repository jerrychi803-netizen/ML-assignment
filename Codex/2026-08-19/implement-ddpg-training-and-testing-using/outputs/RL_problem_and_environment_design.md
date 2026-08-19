# Reinforcement Learning Problem and Environment Design for Fatigue-Aware Short-Video Recommendation

## 1. Problem Framing and Overview

### 1.1 High-level reinforcement-learning problem

Short-video recommendation is a sequential decision-making problem. A platform does not only need to choose a video that is likely to receive an immediate click; it must choose a sequence of videos that maintains attention, encourages positive interactions, and avoids content fatigue. Content fatigue occurs when repeatedly shown, poorly matched, or low-quality videos lead users to skip, dislike, or abandon the session.

The proposed system treats the recommender as a reinforcement-learning (RL) agent and a user session as an environment episode. At each request, the agent observes the current user context and recent viewing history, produces a continuous preference representation, ranks available candidate videos, and recommends the highest-ranked item. The user then generates multiple feedback signals, such as WatchTime, Click, Like, Comment, and Hate. The next recommendation is conditioned on the resulting updated user state.

The project implements a deterministic, offline version of Two-Stage Constrained Actor-Critic (TSCAC) with Deep Deterministic Policy Gradient (DDPG). The main objective is to maximize cumulative WatchTime while preserving positive interactions and limiting negative feedback. This is more appropriate than optimizing a single click-through rate because short-video satisfaction is multi-dimensional and unfolds over a sequence.

### 1.2 RL agent and core components

An RL system consists of the following components.

| Component | Role in this project |
|---|---|
| Agent | The TSCAC-DDPG recommender that selects a continuous user-preference action. |
| Environment | A user interacting with a short-video platform. For offline training, KuaiRand-1K logged user trajectories approximate this environment. |
| Policy | An actor network mapping the current user state to a desired video-preference vector. |
| Value function | A critic network estimating the expected discounted future response after a state-action pair. |
| Action ranking function | Scores each candidate video using the dot product between the actor output and that video's feature vector. |
| Replay dataset | Logged state, action, vector reward, next-state, and terminal-transition tuples created from KuaiRand-1K. |

The actor is the decision-making network. In a state `s_t`, it outputs a deterministic action `a_t = pi_theta(s_t)`. The critic estimates `Q_phi(s_t, a_t)`, the expected discounted cumulative reward after choosing that action. The actor is updated to select actions with larger critic values; the critic is updated using a temporal-difference Bellman target.

### 1.3 Markov Decision Process formulation

The recommendation problem is represented as a Constrained Markov Decision Process (CMDP):

`M = (S, A, P, R, C, rho_0, Gamma)`

where:

- `S` is the state space of user context, recent video history, and recent feedback.
- `A` is the continuous action space. An action is a predicted preference vector, not a raw item ID.
- `P(s_(t+1) | s_t, a_t)` is the unknown user-transition process: after a recommendation, the user watches or skips the video and the state changes.
- `R` is a vector-valued reward function rather than a single scalar reward.
- `C` contains auxiliary-response constraints, such as maintaining Click, Like, and Comment outcomes and not increasing Hate.
- `rho_0` is the distribution of session-start states.
- `Gamma` contains discount factors for each response type.

At each decision point `t`, the process is:

1. Observe user state `s_t`.
2. Produce preference action `a_t = pi(s_t)`.
3. Score candidate video embeddings `v_i` with `score_i = a_t^T v_i`.
4. Recommend the candidate with the highest score.
5. Observe vector reward `r_t` and next state `s_(t+1)`.
6. Continue until the user session ends.

The main objective is to maximize expected cumulative WatchTime:

`max_pi E_pi[sum_t gamma_watch^t * r_watch,t]`

subject to preserving auxiliary outcomes:

`E_pi[V_i(s)] >= C_i`, for Click, Like, and Comment, and minimizing Hate.

### 1.4 Sequential decision-making with an MDP

An MDP is necessary because a recommendation affects more than one immediate response. Recommending a highly engaging video may increase immediate WatchTime but make a user more likely to skip the next few videos if the content becomes repetitive. Conversely, a video that produces a Like or Comment can indicate sustained satisfaction and may improve later engagement.

The agent uses the state to estimate both immediate and long-term consequences. DDPG bootstraps this estimate through the Bellman equation:

`Q(s_t, a_t) = E[r_t + gamma * Q(s_(t+1), pi(s_(t+1)))]`

Thus, the critic learns that the value of a current recommendation includes future user responses, not just the response on the current item.

### 1.5 Model-free and model-based RL

Model-free RL learns a policy and/or value function directly from interaction transitions without explicitly learning a user-transition model. Model-based RL additionally learns or assumes a model of how users transition between states and can use it for planning.

This project uses model-free offline RL for three reasons:

- KuaiRand-1K provides logged transitions and response observations but not a trusted, deployable simulator of all user behavior.
- User behavior is complex, non-stationary, and expensive to model accurately.
- DDPG can learn directly from `(state, action, response, next_state)` records using replay-style mini-batches.

A model-based approach could later be useful for simulated counterfactual evaluation or long-horizon planning, but it would require rigorous validation because simulation error can bias policy learning.

### 1.6 Key learning challenges

**Exploration versus exploitation.** In online systems, a policy must balance serving videos known to work against exploring unfamiliar content. This project is trained offline, so it cannot safely explore. Behavior-cloning regularization keeps proposed continuous actions close to actions represented in the log.

**Sparse rewards.** WatchTime is dense because every impression has a play duration. Likes, Comments, and Hates are much sparser. Combining them into one reward can cause dense WatchTime to overwhelm the sparse signals. TSCAC addresses this with separate critics and separate stage-one auxiliary policies.

**Long-term credit assignment.** A negative reaction may be caused by a recommendation several steps earlier. Critic bootstrapping and a high WatchTime discount factor (`0.99`) allow the model to assign some credit across a trajectory.

**Offline distribution shift.** A deterministic actor can output preference vectors not supported by logged data, causing unreliable critic estimates. The implementation uses behavior-cloning loss, target networks, gradient clipping, checkpoint selection, and a continuous action range constrained by `tanh` to reduce this problem.

**Counterfactual evaluation.** The log reports outcomes only for videos that were actually exposed. It does not reveal what a user would have done for every alternative video. Therefore, reported offline improvements are useful signals, but they are not proof of causal production lift.

### 1.7 Objectives and expected outcomes

The primary learning objective is higher cumulative normalized WatchTime. Secondary objectives are to maintain or improve Click, Like, and Comment responses and to avoid increasing Hate. In a fatigue-aware production extension, short-view/skip rate, repeated-topic exposure, session abandonment, and explicit dislike events should also be modeled as cost signals.

Expected outcomes are:

- A selected `best_tscac_ddpg.pt` checkpoint that improves the primary metric against the BC baseline on validation data.
- Auxiliary metrics that remain near or above baseline thresholds.
- A candidate-ranking policy that can be applied to an online retrieval slate.
- A reproducible offline experiment, not a claim of deployment-ready causal uplift.

### 1.8 Evaluation metrics

The implementation reports the following held-out test metrics.

| Metric | Meaning | Desired direction |
|---|---|---|
| `policy_watch_time` / `bc_watch_time` | Average normalized observed WatchTime after slate ranking. | Higher |
| `policy_click` / `bc_click` | Average observed Click rate. | Higher |
| `policy_like` / `bc_like` | Average observed Like rate. | Higher |
| `policy_comment` / `bc_comment` | Average observed Comment rate. | Higher |
| `policy_hate` / `bc_hate` | Average observed Hate rate. | Lower |
| `constrained_score` | WatchTime after penalties for violating BC-relative auxiliary thresholds. | Higher |
| Critic loss | Bellman-error proxy for critic stability. | Stable/decreasing |
| Actor action MSE | Difference between the actor action and logged actions. | Low but not necessarily zero |
| Best checkpoint step | Stage-two update selected by validation, preventing use of an over-trained final model. | Earlier/later is data-dependent |

The current candidate-slate metric is an offline proxy: contiguous same-user held-out records are treated as a small slate. A more faithful replication of the paper would additionally implement Normalized Capped Importance Sampling (NCIS) with a learned or known behavior-policy probability.

## 2. Environment and Agent Definition

### 2.1 Environment

The target environment is a short-video recommendation platform. A user opens the application, views a sequential feed of videos, produces implicit and explicit feedback, and eventually leaves the session. The production environment is partially observable and expensive to experiment with. For this project, KuaiRand-1K provides offline logged trajectories for approximately one thousand users.

The environment is complex because user response depends on user characteristics, previous videos, previous feedback, video attributes, time, and unavailable latent factors. The model is trained on CPU or GPU with PyTorch. The supplied implementation is practical on CPU for small experiments; GPU is recommended for repeated hyperparameter searches, richer video embeddings, and larger logs.

### 2.2 State space

The current state is a fixed-length numerical vector assembled before each logged interaction. It contains:

- Static user features: live-streamer flag, video-author flag, logarithmically transformed follow count, fan count, friend count, and registration age.
- Recent video history: the feature vectors of the last `H` watched videos; `H = 20` by default to follow the TSCAC paper's history design.
- Recent feedback: prior completion ratio, Like indicator, and Hate indicator.

The current action vector has four dimensions, so the default state dimension is `6 + 20 * 4 + 3 = 89` when all six static user features are available. This is a compact adaptation of the TSCAC paper's richer production state, which also includes candidate features. In a production system, candidate feature embeddings should be passed directly to the ranking layer and richer history features should replace the four simple video attributes.

### 2.3 Action space

The platform ultimately has a discrete action: selecting one video from a candidate slate. DDPG requires continuous actions, so the implementation follows the paper's continuous-preference approach:

1. Each candidate video is represented by a continuous feature vector.
2. The actor outputs a continuous preference vector of the same dimension.
3. Candidate score is the dot product between preference and candidate representation.
4. The highest-scoring video becomes the discrete recommendation.

The current candidate action vector is:

`[log(video_duration), aspect_ratio, music_type, primary_tag]`

It is standardized using training data and scaled to approximately `[-1, 1]`. The primary tag is extracted from KuaiRand's comma-separated tag field. This representation allows a runnable baseline but is not semantically ideal because music and tag IDs are categorical. A stronger implementation should replace categorical IDs with learned embeddings and use all tags, author, upload type, and video statistics.

### 2.4 Reward structure

Each logged transition has a five-dimensional response vector:

`r_t = [completion_ratio, click, like, comment, -hate]`

where:

- `completion_ratio = clip(play_time_ms / duration_ms, 0, 1)` is the WatchTime main reward.
- Click, Like, and Comment are positive auxiliary rewards.
- `-hate` is the anti-Hate auxiliary reward; maximizing it means reducing Hate.

Rewards are standardized from the training split before critic learning because the responses have different scales and frequencies. Raw responses are retained for reporting metrics. This separation prevents sparse signals from being numerically overwhelmed by WatchTime in a shared critic.

### 2.5 Agent definition and learning goal

The learning system contains six neural decision/value components:

- Four stage-one auxiliary actor-critic pairs: Click-DDPG, Like-DDPG, Comment-DDPG, and Anti-Hate-DDPG.
- One stage-two main actor-critic pair: WatchTime-DDPG.
- One supervised behavior-cloning actor used only as an evaluation baseline.

The stage-one agents learn actions that optimize individual auxiliary responses. The stage-two actor learns to maximize WatchTime but is penalized when its action is far from stage-one actions. This soft proximity term is the deterministic counterpart to the paper's policy-distribution constraint.

## 3. RL Algorithm Selection and Model Design

### 3.1 Algorithm choice

The selected method is deterministic TSCAC-DDPG. It combines DDPG with TSCAC's multi-critic and two-stage constrained-policy design.

DDPG is appropriate because the actor output is a continuous preference embedding rather than a direct discrete video ID. A discrete-action method such as DQN would require evaluating every item or an additional large-action retrieval mechanism. Policy-gradient methods with stochastic categorical distributions are also possible, but the continuous action formulation directly supports fast dot-product ranking over a candidate slate.

### 3.2 DDPG architecture

Each DDPG agent has:

- **Actor:** two hidden ReLU layers followed by `tanh`; maps normalized state to a bounded continuous action.
- **Critic:** two hidden ReLU layers; estimates `Q(s, a)` for one response dimension.
- **Target actor and target critic:** slowly updated copies that make Bellman targets more stable.
- **Replay-style mini-batch updates:** random samples are drawn from the offline training transitions.

The critic minimizes:

`L_critic = MSE(Q_phi(s, a), r + gamma * (1-done) * Q_phi_target(s_next, pi_target(s_next)))`

An auxiliary actor maximizes its critic value while remaining close to logged behavior:

`L_aux_actor = -Q_i(s, pi_i(s)) + alpha_BC * ||pi_i(s) - a_logged||^2`

### 3.3 Two-stage constrained actor design

In stage one, each auxiliary actor is optimized independently. This implements the paper's principle that dense and sparse responses should not be collapsed into one critic.

In stage two, the main actor uses the WatchTime critic plus a soft constraint. The implemented loss is:

`L_main_actor = -Q_watch(s, pi_watch(s)) + alpha_BC * MSE(pi_watch(s), a_logged) + lambda * mean_i MSE(pi_watch(s), pi_i(s))`

The final term is a numerically stable Gaussian-kernel-equivalent approximation of the deterministic TSCAC proximity score: actions closer to an auxiliary policy obtain a larger similarity score. `lambda`, exposed as `--constraint-weight`, controls the trade-off. Larger values preserve auxiliary-policy behavior more strongly but can reduce WatchTime.

### 3.4 Model-free rationale

The method is model-free because it does not learn a separate environment transition model. It learns response values directly from observed transitions. This reduces modeling assumptions and matches the available KuaiRand logs. Its limitation is that it cannot safely evaluate actions far from the historical behavior distribution; this is why behavior-cloning regularization and conservative checkpoint selection are used.

### 3.5 Discrete versus continuous action decision

The recommended video itself is discrete, but action selection is performed through a continuous latent preference vector. This is a common scalable approximation for large candidate spaces:

`a_t = actor(s_t)`

`recommended_video = argmax_(v in candidate_slate) a_t^T embedding(v)`

For a stronger model, `embedding(v)` should be learned from full video features or a content encoder. The current four-feature vector is sufficient for an experiment but should not be interpreted as a complete production-quality content embedding.

## 4. Environment Setup and Dynamics

### 4.1 Offline custom environment

The project does not use OpenAI Gym or Unity ML-Agents because KuaiRand-1K is an offline recommendation dataset, not an interactive simulator. Instead, it constructs a custom offline MDP/replay environment from chronological user logs.

Each row in the replay dataset has the form:

`(state_t, action_t, response_vector_t, next_state_t, done_t)`

The episode is one user's chronological viewing sequence. The terminal flag is set to one at the end of that user's sequence. This supplies the state, action, reward, transition, and terminal semantics needed by DDPG.

### 4.2 Reset and step semantics

In a live environment, reset starts a new user session and step executes one recommendation. The offline equivalent is:

- `reset`: select the first recorded state for a user trajectory.
- `step`: consume the next logged transition in that user's chronological sequence.
- `done`: true at the end of the user trajectory.

Because the data is logged, this step process cannot generate a true response for arbitrary recommended alternatives. Instead, training uses observed action-response pairs and evaluation re-ranks small held-out observed candidate windows. This is why live deployment requires cautious A/B testing.

### 4.3 Episodes and training schedule

The complete logged dataset is split chronologically within each user:

- First 70% of interactions: training.
- Next 15%: validation and checkpoint selection.
- Final 15%: untouched final test.

The default training schedule is:

1. Train BC for 1,000 mini-batch updates.
2. Train four stage-one auxiliary DDPG agents for 1,000 updates.
3. Train the stage-two WatchTime DDPG agent for 3,000 updates.
4. Evaluate every 500 stage-two updates.
5. Save `best_tscac_ddpg.pt` when constrained validation score improves.
6. Restore that checkpoint and evaluate it once on the final test split.

The model supports many trajectories and mini-batch updates rather than repeatedly simulating synthetic episodes. A production environment would continually append new interactions to replay storage, periodically retrain, validate, and deploy only reviewed checkpoints.

### 4.4 Recommended experiment command

From the project directory, run:

```powershell
& ".\.venv\Scripts\python.exe" ".\outputs\tscac_ddpg_kuairand.py" `
  --stage1-updates 1000 --stage2-updates 3000 --bc-updates 1000 `
  --constraint-weight 0.5 --aux-ratio 0.98 `
  --output-dir ".\outputs\runs\tscac_ddpg"
```

Use `best_tscac_ddpg.pt` as the selected model. Compare `policy_*` against `bc_*` in `metrics.json`, paying particular attention to whether WatchTime increases while Click, Like, Comment, and Hate remain within the intended constraints.

### 4.5 Limitations and next steps

The current system is a defensible research prototype, not a complete production recommender. The highest-priority improvements are:

1. Replace numeric tag/music IDs with learned multi-tag and content embeddings.
2. Add skip rate, repeated-topic rate, session abandonment, and diversity as explicit fatigue costs.
3. Build real retrieval slates rather than using contiguous logged records as proxy slates.
4. Implement NCIS or doubly robust off-policy evaluation, including behavior-policy probabilities.
5. Report uncertainty intervals through user-level bootstrap resampling.
6. Validate promising policies with a controlled, safety-constrained online A/B test.

These changes are necessary before interpreting offline metric differences as causal evidence that user fatigue has been reduced in a live platform.
