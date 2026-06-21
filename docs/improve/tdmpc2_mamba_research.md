# TD-MPC2 + Mamba 世界模型 — Deep Research 调研报告

> 调研规模: 96 agents, deep research harness (fan-out search → fetch → adversarial verify → synthesize)

> 原始问题: TD-MPC2 (Temporal Difference Learning for Model Predictive Control v2) 的完整技术架构,重点:1) 世界模型结构(encoder、latent dynamics model、reward predictor、critic/q-function 各自的输入输出和网络结构);2) 训练目标函数(latent contrastive ...


## 调研注意点 (Caveats)

Source-access limitation: I could not independently re-fetch arxiv.org/tdmpc2.com during synthesis due to a temporary tool outage; verification rests on verbatim quotes captured during the adversarial-vote phase plus cloned source-code inspection (nicklashansen/tdmpc2 main HEAD 8bbc14e, 2025-05-19). Citation errors in two claims were caught but do not affect substance: (a) claim [20] cited arXiv 2204.11859 (an unrelated paper) instead of the correct TD-MPC1 paper 2203.04955 — content was independently re-verified against the correct source; (b) the secondary 'tdmpc2-jax' port (ShaneFlandermeyer) is a re-implementation, not author-primary, though every load-bearing assertion in it was cross-checked against the official PyTorch repo. Split votes (2-1) appeared on claims [1], [14], [18], [19] — all on secondary sources or v1-era claims — but the disputed details (termination-MLP 'optional' wording, AWR 'style' qualifier, v1 citation metadata) were resolved by primary-source confirmation and do not affect the Mamba-rewrite guidance. The '200x dominance' for the consistency loss is a coefficient ratio (20/0.1), not a raw-loss-magnitude ratio — the actual gradient contribution depends on per-loss magnitudes. Time-sensitivity: the episodic/termination predictor is flagged as an Apr 2025 addition (post-ICLR-2024), so any fork not at main HEAD may lack it; the 317M-param / 104-task / 80-task figures are README-stated targets, not all independently benchmarked in the paper text.


## 核心发现 (Findings)


### Finding 1: World model architecture: TD-MPC2's implicit (decoder-free) world model (called TOLD in v1) has six MLP sub-networks with explicit signatures, all conditioned on a learnable task embedding e (task_dim=96): Encoder z=h(s,e) maps observations to latent space (variable 2-5 layers); Latent dynamics z'=d

- **置信度**: high

- **投票**: 3-0 (claims [5], [10], [19]); 2-1 (claim [1] on six-subnetwork enumeration)

- **证据**: Verified against the official nicklashansen/tdmpc2 repo (world_model.py constructor lines 25-30 builds _encoder/_dynamics/_reward/_termination(if cfg.episodic else None)/_pi/_Qs; __repr__ lines 56-61 labels them Encoder/Dynamics/Reward/Termination/Policy prior/Q-functions) and the ICLR 2024 paper Figure 3 (tdmpc2.txt line 792 architecture dump: dynamics=Sequential(NormedLinear(512+T+A->512,Mish), NormedLinear(512->512,Mish), NormedLinear(512->512,SimNorm))). All five MLPs use 2*[cfg.mlp_dim] hidden dims; _pi output is 2*action_dim (mean+log_std); _Qs uses Ensemble([... for _ in range(cfg.num_q)]) with num_q=5, dropout=0.01. init() lines 38-53 create _detach_Qs and _target_Qs via TensorDictParams+deepcopy; soft_update_target_Q() lines 82-86 does lerp_(tau=0.01) — textbook Polyak. Config: mlp_dim=512, latent_dim=512, task_dim=96, num_q=5, dropout=0.01. The decoder-free property is confirmed by code inspection (no decoder module anywhere); the Dreamer contrast (Dreamer uses recurrent+decoder, TOLD uses deterministic MLPs with no decoder) is stated verbatim in the v1 paper.

- **来源**: https://github.com/nicklashansen/tdmpc2, https://arxiv.org/abs/2310.16828, https://arxiv.org/abs/2204.11859, https://www.tdmpc2.com/


### Finding 2: Latent dynamics d(z,a,e) is a STATELESS single-step feedforward 3-layer MLP (NormedLinear->Mish x2 -> SimNorm output), input concat([z, a, task]) of dim (latent_dim+action_dim+task_dim)=512+A+96, output dim latent_dim=512. There is NO recurrent cell, NO hidden state, NO GRU/LSTM/Transformer. Multi-s

- **置信度**: high

- **投票**: 3-0 (claims [0], [6], [12], [15] all unanimous)

- **证据**: Verified against the official nicklashansen/tdmpc2 repo (world_model.py:26 self._dynamics = layers.mlp(cfg.latent_dim+cfg.action_dim+cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg)); next() lines 114-121: z=cat([z,a],dim=-1); return self._dynamics(z) — no hidden state). layers.py mlp() builds dims=[in,mlp_dim,mlp_dim,out], appends 2 NormedLinear(Mish) + 1 NormedLinear(SimNorm) = exactly 3 layers. Multi-step unrolling confirmed in _update: 'for t,(_action,_next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))): z = self.model.next(z, _action, task); consistency_loss += F.mse_loss(z, _next_z)*self.cfg.rho**t; zs[t+1]=z'; same loop pattern in _estimate_value and _plan. The v1 paper (arXiv 2203.04955) states verbatim: 'recurrent predictions are made entirely in latent space from states z_i=h(s_i), z_{i+1}=d(z_i,a_i), ..., z_{i+H}=d(z_{i+H-1},a_{i+H-1}) such that only the first observation s_i is encoded using h and gradients from all three terms are back-propagated through time.' No nn.GRU/LSTM/RNN/MultiheadAttention anywhere in layers.py. The figure caption's word 'recurrently' refers to the rollout loop feeding z back in, NOT a recurrent cell — confirmed by released code being dispositive. The JAX port (ShaneFlandermeyer/tdmpc2-jax) independently confirms the same structure: dynamics_module=nn.Sequential([NormedLinear(Mish),NormedLinear(Mish),NormedLinear(None)]) with next(z,a)=simnorm(dynamics(cat([z,a]))).

- **来源**: https://github.com/nicklashansen/tdmpc2, https://arxiv.org/abs/2310.16828, https://arxiv.org/abs/2203.04955, https://github.com/ShaneFlandermeyer/tdmpc2-jax


### Finding 3: Training objectives: h,d,R,Q are jointly optimized over horizon H with a weighted loss (the policy prior p uses a SEPARATE optimizer). Total loss = consistency_coef*consistency + reward_coef*reward + value_coef*value + termination_coef*termination_loss, with consistency_coef=20, reward_coef=0.1, val

- **置信度**: high

- **投票**: 3-0 (claims [2], [7], [13]); 2-1 (claim [18] on AWR-style policy loss)

- **证据**: Verified against official repo (tdmpc2.py _update): 'consistency_loss += F.mse_loss(z, _next_z)*self.cfg.rho**t' where _next_z=encode(obs[1:]) under torch.no_grad() (stop-grad sg); 'reward_loss += math.soft_ce(rew_pred_unbind, rew_unbind, cfg).mean()*self.cfg.rho**t'; 'value_loss += math.soft_ce(qs_unbind_unbind, td_targets_unbind, cfg).mean()*self.cfg.rho**t'; 'total_loss = consistency_coef*consistency + reward_coef*reward + termination_coef*termination_loss + value_coef*value'. _td_target returns 'reward + discount*(1-terminated)*self.model.Q(next_z, action, task, return_type=min, target=True)' where action,_=self.model.pi(next_z,task) (policy prior p(z'_t)) and target=True selects EMA target Q. Config: consistency_coef=20, reward_coef=0.1, value_coef=0.1, termination_coef=1, rho=0.5. soft_ce (math.py:6-10) = -(target*log_softmax(pred)).sum(-1) = cross-entropy with soft two-hot targets. Optimizer covers _encoder,_dynamics,_reward,_Qs (NOT _pi); policy uses separate pi_optim (tdmpc2.py:221-227: pi_loss uses return_type='avg' = MEAN of 2 subsampled Qs; rho=torch.pow(cfg.rho, arange(len(qs))); pi_loss = (-(entropy_coef*scaled_entropy + qs).mean(dim=(1,2))*rho).mean()). RunningScale (scale.py:12,41,47): _percentiles=[5,95]; value=clamp(percentiles[1]-percentiles[0], min=1.); return x/self.value. The v1 paper (arXiv 2203.04955 Eq 8-10) confirms the same three-loss joint objective with c1,c2,c3 coefficients and BPTT — v2 adds the multi-step rho^t weighting and switches consistenc
...
- **来源**: https://github.com/nicklashansen/tdmpc2, https://arxiv.org/abs/2310.16828, https://arxiv.org/abs/2203.04955, https://github.com/ShaneFlandermeyer/tdmpc2-jax


### Finding 4: Distributional reward and value heads: both the reward MLP R and the Q-ensemble heads are DISTRIBUTIONAL (discrete regression), outputting num_bins=101 logits over a symlog-scaled support in [vmin=-10, vmax=+10], decoded via two_hot_inv (softmax over bins + symexp) and trained with soft (two-hot) cr

- **置信度**: high

- **投票**: 3-0 (claims [4], [16] both unanimous)

- **证据**: Verified against both the official nicklashansen/tdmpc2 repo and the JAX port. Config: num_bins=101, vmin=-10, vmax=+10. math.py two_hot (lines 59-84): applies symlog then clamps to [vmin,vmax], bins=linspace(vmin,vmax,num_bins) => symlog-spaced in original space; two_hot_inv applies symexp. Both _reward (world_model.py:27) and _Qs (world_model.py:30) output max(cfg.num_bins,1)=101 dims. soft_ce (math.py:6-10): -(target*log_softmax(pred)).sum(-1) used for both reward_loss (tdmpc2.py:289) and value_loss (tdmpc2.py:291). TD target (tdmpc2.py:258): 'return reward + discount*(1-terminated)*self.model.Q(next_z, action, task, return_type=min, target=True)'. world_model.py:212-216 (return_type='min'): qidx=torch.randperm(num_q)[:2]; Q=two_hot_inv(out[qidx]); return Q.min(0).values. world_model.py:201-202 (return_type='avg' for MPC/policy): average of two randomly subsampled Q-values. JAX port (tdmpc2.py:342-349) confirms: 'inds=jax.random.choice(ensemble_key, arange(0, num_value_nets), shape=(2,), replace=False); Q=Qs[inds].min(axis=0); td_targets=rewards+(1-terminated)*self.discount*Q'. This is a v1->v2 change: v1 used continuous regression (unstable for large rewards); v2 uses discrete regression (stable).

- **来源**: https://github.com/nicklashansen/tdmpc2, https://github.com/ShaneFlandermeyer/tdmpc2-jax


### Finding 5: Inference-time MPC planning: TD-MPC2 uses Model Predictive Path Integral (MPPI), a derivative-free optimizer over the learned latent world model. At each step it samples 512 action sequences of horizon=3 (24 of which are policy-prior-seeded via num_pi_trajs=24), refines over 6 iterations, evaluates 

- **置信度**: high

- **投票**: 3-0 (claims [3], [8], [17]); 2-1 (claim [14] on v1 MPPI formulation)

- **证据**: Verified against official nicklashansen/tdmpc2 repo (tdmpc2.py _plan and _estimate_value). Config: mpc=true, iterations=6, num_samples=512, num_elites=64, horizon=3, temperature=0.5, num_pi_trajs=24. _plan (line 173 comment '# Iterate MPPI'): for _ in range(iterations): actions=mean.unsqueeze(1)+std.unsqueeze(1)*r (r~N(0,I) = diagonal Gaussian); value=_estimate_value(z,actions,task).nan_to_num(0); elite_idxs=torch.topk(value.squeeze(1), num_elites).indices; elite_value,elite_actions=value[elite_idxs],actions[:,elite_idxs]; score=torch.exp(temperature*(elite_value-max_value)); mean=(score.unsqueeze(0)*elite_actions).sum(dim=1)/(score.sum(0)+1e-9) (NO momentum mixing); std=sqrt(weighted variance); rand_idx=math.gumbel_softmax_sample(score.squeeze(1)); actions=torch.index_select(elite_actions,1,rand_idx). _estimate_value (lines 124-137): for t in range(horizon): reward=two_hot_inv(self.model.reward(z,actions[t],task),cfg); z=self.model.next(z,actions[t],task); G=G+discount*(1-term)*reward; discount*=self.cfg.discount; return G+discount*(1-term)*self.model.Q(z,action,task,return_type='avg'). Lines 155-171: first num_pi_trajs samples from policy prior pi(_z,task) rolled through model.next. Lines 167-168: mean[:-1]=self._prev_mean[1:] (warm-start shifted by 1). Momentum removal confirmed by direct code diff: v1 (nicklashansen/tdmpc src/algorithm/tdmpc.py:141) 'mean,std=cfg.momentum*mean+(1-cfg.momentum)*_mean,_std' vs v2 direct assignment. (iterations becomes 8 for action_dim>=20, 
...
- **来源**: https://github.com/nicklashansen/tdmpc2, https://arxiv.org/abs/2310.16828, https://arxiv.org/abs/2203.04955, https://github.com/ShaneFlandermeyer/tdmpc2-jax


### Finding 6: TD-MPC2 vs TD-MPC1 improvements (5 concrete changes): (a) SimNorm latent normalization (softmax into L simplices) + LayerNorm + Mish, replacing v1's unconstrained latent + ELU (which caused exploding gradients); (b) ensemble of 5 Q-functions with 1% Dropout and min-of-2-sampled TD-targets, vs v1's 2

- **置信度**: high

- **投票**: 3-0 (claims [9], [11]); claim [4] (3-0) adds single-hyperparameter/317M/episodic details

- **证据**: Verified verbatim against the ICLR 2024 paper (arXiv 2310.16828): 'All components of TD-MPC 2 are MLPs with LayerNorm ... and Mish ... We apply SimNorm normalization to the latent state z ... We train an ensemble of Q-functions (5 by default) and additionally apply 1% Dropout ... TD-targets are computed as the minimum of two randomly subsampled Q-functions. In contrast, TD-MPC is implemented as MLPs without LayerNorm, and instead uses ELU ... TD-MPC does not constrain the latent state at all, which in some instances leads to exploding gradients ... TD-MPC learns only 2 Q-functions and does not use Dropout ... TD-MPC 2 uses discrete regression (soft cross-entropy) of rewards and values in a log-transformed space ... TD-MPC uses continuous regression which leads to training instabilities ... The policy prior of TD-MPC 2 is trained with maximum entropy RL ... whereas the policy prior of TD-MPC is trained as a deterministic policy with Gaussian noise ... TD-MPC 2 removes momentum in MPPI ... and replaces prioritized experience replay sampling ... with uniform sampling.' README confirms: '104 continuous control tasks spanning multiple domains, with a single set of hyperparameters ... training a single 317M parameter agent to perform 80 tasks ... support for episodic RL (tasks with terminations) ... enabled with episodic=true' (Apr 2025 announcement). Abstract (tdmpc2.com): 'we present TD-MPC2: a series of improvements upon the TD-MPC algorithm.' Momentum-removal independently conf
...
- **来源**: https://arxiv.org/abs/2310.16828, https://github.com/nicklashansen/tdmpc2, https://www.tdmpc2.com/


### Finding 7: Design rationale and decoder-free property: TD-MPC2 is a model-based RL (MBRL) algorithm whose world model is IMPLICIT and DECODER-FREE, and it performs local trajectory optimization in LATENT space (not pixel/observation space) at inference. The core design fuses short-horizon MPC planning (local o

- **置信度**: high

- **投票**: 3-0 (claims [10], [20] both unanimous); claim [20] citation metadata error caught and corrected to 2203.04955

- **证据**: Verified against the official tdmpc2.com site: 'TD-MPC is a model-based reinforcement learning (MBRL) algorithm that performs local trajectory optimization in the latent space of a learned implicit (decoder-free) world model. In this work, we present TD-MPC2: a series of improvements upon the TD-MPC algorithm.' The v1 paper (arXiv 2203.04955, the correct source — note claim [20] erroneously cited 2204.11859 which is an unrelated paper) states verbatim in its abstract: 'In this work, we combine the strengths of model-free and model-based methods. We use a learned task-oriented latent dynamics model for local trajectory optimization over a short horizon, and use a learned terminal value function to estimate long-term return, both of which are learned jointly by temporal difference learning.' Code-level verification: world_model.py:26 constructs self._dynamics=layers.mlp(...); next() is a single MLP forward z=cat([z,a]); return self._dynamics(z) — predicts the next LATENT z from (z,a), with NO decoder module anywhere in layers.py. MPC scoring (_estimate_value) operates entirely on latent z (model.next, model.reward, model.Q all take latent z as input); only the initial encode(obs) touches observation space. A local UniLab design doc (轻量Transformer-Mamba混合通用运控模型架构设计.md) that loosely called TD-MPC2 'reconstruction+KL / decoder-predicts-observations' was confirmed to be non-authoritative team terminology contradicting both the paper and the code (no decoder found); the team plans t
...
- **来源**: https://www.tdmpc2.com/, https://arxiv.org/abs/2310.16828, https://arxiv.org/abs/2203.04955, https://arxiv.org/abs/2204.11859
