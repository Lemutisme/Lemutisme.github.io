---
title: "DR-SAC: Robust to the Worst Plausible World, Cheaply"
description: "The bottleneck in distributionally robust RL is not defining the worst-case transition model, but making the worst-case soft Bellman backup computable: KL duality, decomposable functional optimization, and a conditional transition VAE turn an infinite-dimensional measure problem into a five-step update inside Soft Actor-Critic."
publishedAt: 2026-11-07
tags:
  - Distributionally Robust Optimization
  - Offline Reinforcement Learning
  - Soft Actor-Critic
  - Continuous Control
draft: false
---

**DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty** · **ICLR 2026**

Mingxuan Cui\*, **Duo Zhou**\*<sup>†</sup>, Yuxuan Han, Grani A. Hanasusanto, Qiong Wang, Huan Zhang, Zhengyuan Zhou

[OpenReview](https://openreview.net/forum?id=a19MA0ksbc) · [arXiv](https://arxiv.org/abs/2506.12622) · [PDF](https://arxiv.org/pdf/2506.12622) · [Code](https://github.com/Lemutisme/DR-SAC) · [Project page](/projects/distributionally-robust-control/)

In distributionally robust reinforcement learning (DR-RL), writing down the worst-case objective is straightforward: replace the single transition kernel of a Markov Decision Process with an ambiguity set of plausible kernels around the nominal dynamics, and evaluate the policy against the worst transition model in that set. The real obstacle is computational. Inside an actor-critic loop over continuous state and action spaces, a worst-case Bellman backup requires minimizing expected next-state value over an infinite-dimensional family of probability measures at every state-action pair—and doing so from a static offline dataset without ever querying a simulator.

DR-SAC makes **the worst-case soft Bellman backup computable at actor-critic scale**. Kullback–Leibler (KL) strong duality first collapses the infinite-dimensional inner infimum over transition kernels into a one-dimensional concave search over a scalar temperature $\beta \ge 0$ evaluated strictly under the nominal transition model. Decomposable functional optimization then replaces the per-sample scalar searches across the replay buffer with a single shared dual network trained in five gradient steps per critic update, while a conditional Variational Autoencoder (VAE) supplies the local transition samples needed to prevent the nonlinear KL dual from degenerating on continuous offline data. [Paper, Sections 1 and 3](https://arxiv.org/abs/2506.12622).

<figure>
  <img src="/blog/dr-sac/dual-backup-pipeline.svg" alt="Diagram of the three reductions in DR-SAC: (1) KL strong duality reduces the infinite-dimensional measure infimum to a 1D scalar search over beta under the nominal kernel; (2) L1 decomposable interchange replaces per-(s,a) SciPy solves with a shared dual network g_eta updated in 5 gradient steps; (3) a conditional transition VAE draws m next-state samples per (s,a) to prevent single-sample log-exp cancellation.">
  <figcaption>The three obstructions between the Robust MDP objective and an offline continuous-action actor-critic algorithm, paired with the exact reduction DR-SAC uses at each stage.</figcaption>
</figure>

## 1. Why nominal-model offline RL breaks under shift

Consider a discounted continuous-state, continuous-action Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, R, P, \gamma)$ with bounded reward $r = R(s,a) \in [0, R_{\max}]$, transition kernel $p_{s,a} = P(\cdot \mid s,a) \in \Delta(\mathcal{S})$, discount factor $\gamma \in [0, 1)$, and stochastic policy $\pi : \mathcal{S} \to \Delta(\mathcal{A})$. Maximum entropy reinforcement learning augments reward with policy entropy $\mathcal{H}(\pi(s)) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[-\log \pi(a \mid s)]$ weighted by temperature $\alpha \ge 0$, defining the soft value function

$$
V_{\mathcal{M}}^\pi(s) = \mathbb{E}\!\left[\, \sum_{t=1}^{\infty} \gamma^{t-1}\!\left(r_t + \alpha \, \mathcal{H}(\pi(s_t))\right) \;\mid\; \pi,\, s_1 = s \,\right]
$$

and soft action-value function $Q_{\mathcal{M}}^\pi(s,a)$. Soft Actor-Critic (SAC) alternates policy evaluation under the nominal soft Bellman operator

$$
\mathcal{T}^\pi Q(s,a) = \mathbb{E}[r] + \gamma \, \mathbb{E}_{s' \sim p_{s,a},\, a' \sim \pi(\cdot \mid s')}\!\left[Q(s',a') - \alpha \log \pi(a' \mid s')\right]
$$

with Boltzmann policy improvement $\pi_{k+1} = \operatorname{argmin}_{\pi \in \Pi} D_{\mathrm{KL}}\!\left(\pi(\cdot \mid s) \,\Vert\, \exp(\frac{1}{\alpha}Q_{\mathcal{M}}^{\pi_k}(s,\cdot)) / Z(s)\right)$. [Paper, Section 2.1, Equations (1)–(5)](https://arxiv.org/abs/2506.12622).

In offline RL, the agent has access only to a static dataset $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^N$ of $N = 10^6$ transitions drawn from a nominal environment $P^0 = \{p_{s,a}^0\}_{(s,a)\in\mathcal{S}\times\mathcal{A}}$ under a behavior policy distribution $\mu$. Because $\mathcal{T}^\pi$ trusts $p_{s,a}^0$ as exact, the critic and actor tune actions to the nominal dynamics. When deployment shifts the physical plant—changing pendulum length or mass, joint damping or stiffness in MuJoCo locomotion, thruster power or wind on a lander, or injecting sensor and actuator noise—trajectories visit transitions where the nominal value estimate was overly optimistic, and return degrades steeply. [Paper, Sections 1, 2.2, and 4.2](https://arxiv.org/abs/2506.12622).

Prior distributionally robust RL methods do not close this gap for continuous control:

- **Tabular DR-RL** relies on enumerating state-action pairs or empirical transition frequencies, which is impossible when $\mathcal{S}$ and $\mathcal{A}$ are continuous.
- **Robust Fitted Q-Iteration (RFQI)** extends offline DR-RL beyond tabular settings using a Total Variation (TV) ambiguity set on top of Fitted Q-Iteration (FQI) and Batch-Constrained Q-learning (BCQ). However, value-based FQI learns deterministic policies, scales poorly in continuous action spaces, and is brittle to dataset action coverage: on both `Pendulum` and `HalfCheetah`, FQI and RFQI fail even in the unperturbed nominal environment when trained on $\epsilon$-greedy offline datasets. [Paper, Sections 1 and 4.2](https://arxiv.org/abs/2506.12622).

## 2. The distributionally robust soft Bellman operator

To protect against transition mismatch without sacrificing maximum-entropy actor-critic learning, DR-SAC formulates a Robust Markov Decision Process $\mathcal{M}_\delta = (\mathcal{S}, \mathcal{A}, R, \mathcal{P}(\delta), \gamma)$ over an $(s,a)$-rectangular ambiguity set $\mathcal{P}(\delta) = \prod_{(s,a)\in\mathcal{S}\times\mathcal{A}} \mathcal{P}_{s,a}(\delta)$. Around each nominal transition kernel $p_{s,a}^0$, the plausible deployment kernels lie in a KL-divergence ball of radius $\delta > 0$:

$$
\mathcal{P}_{s,a}(\delta) := \left\{\, p_{s,a} \in \Delta(\mathcal{S}) : D_{\mathrm{KL}}\!\left(p_{s,a} \,\Vert\, p_{s,a}^0\right) \le \delta \,\right\},
\qquad
D_{\mathrm{KL}}(P \,\Vert\, Q) = \mathbb{E}_P\!\left[\log \frac{P(x)}{Q(x)}\right].
$$

The distributionally robust soft value and soft $Q$-function evaluate a policy $\pi$ against the worst transition kernel $p \in \mathcal{P}(\delta)$:

$$
V_{\mathcal{M}_\delta}^\pi(s) = \inf_{p \in \mathcal{P}(\delta)} \mathbb{E}_p\!\left[\, \sum_{t=1}^{\infty} \gamma^{t-1}\!\left(r_t + \alpha \, \mathcal{H}(\pi(s_t))\right) \;\mid\; \pi,\, s_1 = s \,\right],
$$

$$
Q_{\mathcal{M}_\delta}^\pi(s,a) = \inf_{p \in \mathcal{P}(\delta)} \mathbb{E}_p\!\left[\, r_1 + \sum_{t=2}^{\infty} \gamma^{t-1}\!\left(r_t + \alpha \, \mathcal{H}(\pi(s_t))\right) \;\mid\; \pi,\, s_1 = s,\, a_1 = a \,\right].
$$

By $(s,a)$-rectangularity, dynamic programming holds and $Q_{\mathcal{M}_\delta}^\pi$ is the fixed point of the **distributionally robust soft Bellman operator**:

$$
\mathcal{T}_\delta^\pi Q(s,a) := \mathbb{E}[r] + \gamma \inf_{p_{s,a} \in \mathcal{P}_{s,a}(\delta)} \mathbb{E}_{s' \sim p_{s,a},\, a' \sim \pi(\cdot \mid s')}\!\left[Q(s',a') - \alpha \log \pi(a' \mid s')\right].
$$

[Paper, Section 2.2, Equations (6)–(9), and Section 3.1, Equation (10)](https://arxiv.org/abs/2506.12622).

Equation (10) isolates the algorithmic difficulty. Every application of $\mathcal{T}_\delta^\pi$ requires solving an inner minimization over probability measures $p_{s,a} \in \Delta(\mathcal{S})$ constrained by $D_{\mathrm{KL}}(p_{s,a} \,\Vert\, p_{s,a}^0) \le \delta$. Over a continuous state space $\mathcal{S}$, that inner problem is infinite-dimensional at each $(s,a)$.

## 3. Duality and functional optimization turn an inner kernel search into a shared scalar update

### 3.1 KL strong duality eliminates the adversarial kernel

Define the soft state value under the current critic and policy:

$$
V(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\!\left[Q(s,a) - \alpha \log \pi(a \mid s)\right].
$$

Writing $\inf_{p_{s,a} \in \mathcal{P}_{s,a}(\delta)} \mathbb{E}_{p_{s,a}}[V(s')] = -\sup_{p_{s,a} \in \mathcal{P}_{s,a}(\delta)} \mathbb{E}_{p_{s,a}}[-V(s')]$ and applying strong duality for KL-constrained worst-case expectations (Hu & Hong, 2013, Lemma B.1; valid because bounded rewards and $\gamma \in [0,1)$ keep $V(s')$ bounded with a finite moment generating function around zero), **Proposition 3.3** rewrites the primal operator in exact dual form:

$$
\mathcal{T}_\delta^\pi Q(s,a) = \mathbb{E}[r] + \gamma \sup_{\beta \ge 0} \left\{ -\beta \log\!\left(\mathbb{E}_{s' \sim p_{s,a}^0}\!\left[\exp\!\left(-\frac{V(s')}{\beta}\right)\right]\right) - \beta \delta \right\}.
$$

[Paper, Proposition 3.3, Equations (11)–(12), and Appendix B.1](https://arxiv.org/abs/2506.12622).

This identity accomplishes two things at once:

1. **The inner optimization drops from infinite dimensions to one scalar.** At each $(s,a)$, the adversarial distribution disappears; the solver searches only over a non-negative dual temperature $\beta \ge 0$.
2. **Expectations are taken strictly under the nominal transition kernel $p_{s,a}^0$.** Rather than constructing perturbed transition kernels in $\mathcal{P}(\delta)$, the backup reweights nominal next states exponentially via $\exp(-V(s')/\beta)$, placing higher effective mass on transitions with low continuation value $V(s')$.

Despite the nonlinear log-moment-generating-function form, the operator preserves the exact fixed-point geometry needed for actor-critic learning:

- **$\gamma$-contraction (Proposition 3.4):** For any two bounded critics $Q_1, Q_2$ with soft values $V_1, V_2$, monotonicity of the log-exponential functional $F_\beta(V) = -\beta \log \mathbb{E}_{p_{s,a}^0}[\exp(-V(s')/\beta)] - \beta\delta$ ensures $\vert F_\beta(V_1) - F_\beta(V_2)\vert \le \Vert V_1 - V_2 \Vert_\infty$ for every $\beta > 0$, while at the boundary $\beta^\star = 0$ (which occurs if and only if $p_{s,a}^0$ places sufficient mass $\kappa > 0$ with $\log \kappa + \delta \ge 0$ on $\operatorname{ess\,inf}_{s'\sim p_{s,a}^0} V(s')$, by Lemma B.2) the objective reduces to the essential infimum. Across all three optimality cases ($\beta_1^\star = \beta_2^\star = 0$; one zero and one positive; both positive), $\Vert \mathcal{T}_\delta^\pi Q_1 - \mathcal{T}_\delta^\pi Q_2 \Vert_\infty \le \gamma \Vert Q_1 - Q_2 \Vert_\infty$. Iterative application $Q^{k+1} = \mathcal{T}_\delta^\pi Q^k$ therefore converges globally to $Q_{\mathcal{M}_\delta}^\pi$. [Paper, Proposition 3.4 and Appendix B.2](https://arxiv.org/abs/2506.12622).
- **Monotonic policy improvement and convergence (Proposition 3.5 and Theorem 3.6):** Updating the policy via the KL projection

$$
\pi_{k+1} = \operatorname{argmin}_{\pi \in \Pi} D_{\mathrm{KL}}\!\left(\pi(\cdot \mid s) \;\Vert\; \frac{\exp\!\left(\frac{1}{\alpha} Q_{\mathcal{M}_\delta}^{\pi_k}(s,\cdot)\right)}{Z^{\pi_k}(s)}\right)
$$

guarantees $Q_{\mathcal{M}_\delta}^{\pi_{k+1}}(s,a) \ge Q_{\mathcal{M}_\delta}^{\pi_k}(s,a)$ for all $(s,a) \in \mathcal{S}\times\mathcal{A}$ under $\vert \mathcal{A}\vert < \infty$, and alternating DR soft policy evaluation with DR soft policy improvement converges to the distributionally robust optimal policy $\pi^\star$. [Paper, Section 3.1, Equations (13)–(14), and Appendices B.3–B.4](https://arxiv.org/abs/2506.12622).

### 3.2 Decomposable functional optimization removes the per-sample inner loop

Collapsing the inner problem to a scalar $\beta \ge 0$ makes a single backup tractable, but a naive implementation (`DR-SAC-Accurate`) must still run a separate 1D numerical optimization for every $(s,a)$ pair in every mini-batch. On `Pendulum`, solving Equation (11) per sample via SciPy takes **260 minutes** for 8,000 training steps, compared with **1.7 minutes** for 10,000 steps of standard SAC—over $150\times$ slower. [Paper, Section 4.3.1 and Appendix C.3.1, Table 5](https://arxiv.org/abs/2506.12622).

DR-SAC eliminates the per-sample solver by pulling the supremum over $\beta$ outside the dataset expectation. Let

$$
f\!\left((s,a), \beta\right) := -\beta \log\!\left(\mathbb{E}_{s' \sim p_{s,a}^0}\!\left[\exp\!\left(-\frac{V(s')}{\beta}\right)\right]\right) - \beta \delta.
$$

Because $V(s') \in [0,\, V_{\max}]$ with $V_{\max} = (R_{\max} + \alpha \log \vert\mathcal{A}\vert)/(1-\gamma)$ and the primal value satisfies $f((s,a), \beta^\star) = \inf_{p_{s,a} \in \mathcal{P}_{s,a}(\delta)} \mathbb{E}[V(s')] \ge 0$, any $\beta > V_{\max}/\delta$ yields $f((s,a),\beta) \le V_{\max} - \beta\delta < 0$ and can never be optimal. The pointwise optimal multiplier $\beta^\star(s,a)$ is therefore uniformly bounded in the compact interval $[0, V_{\max}/\delta]$. Defining the admissible function space on $(\mathcal{S}\times\mathcal{A}, \Sigma(\mathcal{S}\times\mathcal{A}), \mu)$,

$$
\mathcal{G} := \left\{\, g \in L^1(\mathcal{S}\times\mathcal{A}, \Sigma(\mathcal{S}\times\mathcal{A}), \mu) : g(s,a) \in \left[0,\; \frac{R_{\max} + \alpha \log \vert\mathcal{A}\vert}{(1-\gamma)\delta}\right],\; \forall (s,a) \in \mathcal{S}\times\mathcal{A} \,\right\},
$$

$f((s,a),\beta)$ is a finite-valued normal integrand (measurable in $(s,a)$ and continuous in $\beta$). Applying Rockafellar and Wets's interchange of minimization and integration in decomposable spaces (Lemma B.4), **Proposition 3.7** establishes the exact functional identity:

$$
\mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\sup_{\beta \ge 0} f\!\left((s,a), \beta\right)\right] = \sup_{g \in \mathcal{G}} \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\, f\!\left((s,a), g(s,a)\right) \,\right].
$$

[Paper, Section 3.2, Equations (15)–(17), and Appendix B.5](https://arxiv.org/abs/2506.12622).

Substituting a parameterized neural network $g \in \mathcal{G}_\eta$ for the pointwise scalar $\beta$ defines the **functional DR soft Bellman operator**:

$$
\mathcal{T}_{\delta, g}^\pi Q(s,a) := \mathbb{E}[r] + \gamma \left\{ -g(s,a) \log\!\left(\mathbb{E}_{s' \sim p_{s,a}^0}\!\left[\exp\!\left(-\frac{V(s')}{g(s,a)}\right)\right]\right) - g(s,a)\delta \right\}.
$$

Instead of solving $\vert\mathcal{D}\vert$ independent scalar optimizations per update, DR-SAC updates $\eta$ jointly across the mini-batch to maximize $\mathbb{E}_{(s,a)\sim\mathcal{D}}[\widetilde{f}((s,a), g(s,a))]$ in just **5 gradient steps**. [Paper, Section 3.2, Equation (18), and Section 4.3.1](https://arxiv.org/abs/2506.12622).

## 4. Making it work offline: the double-sampling trap and generative transitions

### 4.1 Why naive offline ERM collapses under KL duality

In offline RL, $p_{s,a}^0$ is unknown; we only observe transitions $(s_i, a_i, r_i, s_i') \in \mathcal{D}$. Under a Total Variation ambiguity set (as in RFQI), the dual Bellman operator is *linear* in $p_{s,a}^0$, so standard single-sample Empirical Risk Minimization (ERM) over $\mathcal{D}$ is unbiased.

Under a KL ambiguity set, the dual objective $f((s,a), g(s,a))$ applies a nonlinear $-\log(\cdot)$ *outside* the inner expectation $\mathbb{E}_{s' \sim p_{s,a}^0}[\exp(-V(s')/g(s,a))]$. Attempting to estimate both the outer dataset expectation and the inner transition expectation directly from $\mathcal{D}$ triggers the classic **double-sampling issue** (Baird, 1995):

- **Disjoint conditional sample sets are empty in continuous spaces.** Splitting the dataset into $\mathcal{D}_{\mathrm{out}}$ and $\mathcal{D}_{\mathrm{in}}$ and forming the conditional sample set $\mathcal{D}_{(s,a)} = \{(\bar{s}, \bar{a}, \bar{s}') \in \mathcal{D} : (\bar{s}, \bar{a}) = (s,a)\}$ yields the plug-in estimator:

$$
\widehat{J}(g) := \frac{1}{\vert\mathcal{D}_{\mathrm{out}}\vert} \sum_{(s,a,s') \in \mathcal{D}_{\mathrm{out}}} \left[ -g(s,a) \log\!\left(\frac{1}{\vert\mathcal{D}_{(s,a)}\vert} \sum_{(\bar{s},\bar{a},\bar{s}') \in \mathcal{D}_{(s,a)}} \exp\!\left(-\frac{V(\bar{s}')}{g(s,a)}\right)\right) - g(s,a)\delta \right].
$$

- **Single-sample evaluation cancels the logarithm and exponential.** In continuous state and action spaces, the probability of visiting the exact same $(s,a)$ pair twice in a finite dataset is zero, so $\mathcal{D}_{(s,a)}$ is either empty (if split disjointly) or contains only the single recorded transition $s_i'$ ($m_i = 1$). With a single sample $s_i'$ inside the logarithm, $\log$ and $\exp$ cancel identically:

$$
-g(s_i, a_i) \log\!\left(\exp\!\left(-\frac{V(s_i')}{g(s_i, a_i)}\right)\right) - g(s_i, a_i)\delta = V(s_i') - g(s_i, a_i)\delta.
$$

Because $g(s_i, a_i) \ge 0$ and $\delta > 0$, maximizing $V(s_i') - g(s_i, a_i)\delta$ drives the dual multiplier to $g^\star(s_i, a_i) = 0$ everywhere, **collapsing the robust operator back into the non-robust Bellman backup**. As Equation (25) of the paper shows, the exact per-sample operator $\mathcal{T}_\delta^\pi$ suffers from the identical cancellation if estimated with one next state per $(s,a)$. [Paper, Section 3.2 and Appendix A.1, Equations (23)–(25)](https://arxiv.org/abs/2506.12622).

### 4.2 Conditional transition VAE and the explicit V-network

To evaluate the inner expectation without a simulator or repeated state-action visits, DR-SAC trains a conditional Variational Autoencoder (VAE) parameterized by $\varphi$ on $(s, a, s') \in \mathcal{D}$ to approximate $p_{s,a}^0$. With latent prior $p(z) = \mathcal{N}(0, I)$, encoder $q_\varphi(z \mid s, a, s')$, and decoder reconstructing $\hat{s}'$ from $(s, a, z)$, the VAE minimizes the ELBO loss:

$$
J_{\mathrm{VAE}}(\varphi) = \mathbb{E}_{q_\varphi(z \mid s,a,s')}\!\left[\Vert s' - \hat{s}' \Vert^2\right] + D_{\mathrm{KL}}\!\left(q_\varphi(z \mid s,a,s') \,\Vert\, \mathcal{N}(0, I)\right).
$$

At each gradient step, for every $(s,a)$ in the mini-batch, the decoder draws $m$ synthetic next states $\{\tilde{s}_i'\}_{i=1}^m$ to form the local empirical measure $\tilde{p}_{s,a}^0$, yielding the **empirical DR soft Bellman operator**:

$$
\widetilde{\mathcal{T}}_{\delta, g}^\pi Q(s,a) := \mathbb{E}[r] + \gamma \, \widetilde{f}\!\left((s,a), g(s,a)\right),
\qquad
\widetilde{f}\!\left((s,a), \beta\right) = -\beta \log\!\left(\frac{1}{m}\sum_{i=1}^m \exp\!\left(-\frac{V_{\bar{\psi}}(\tilde{s}_i')}{\beta}\right)\right) - \beta \delta.
$$

[Paper, Section 3.2, Equations (19)–(20), and Appendix A.2, Equations (27) and (30)](https://arxiv.org/abs/2506.12622).

Two architectural details in Section 3.3 and Appendix C.3.3 make this empirical operator stable on offline continuous-control datasets:

1. **Retaining an explicit state-value network $V_\psi(s)$ (SAC-v1) rather than dropping it (SAC-v2).** Later online SAC variants remove $V_\psi$ and estimate target values directly from $Q_{\bar{\theta}}(s', a')$. In offline learning, however, removing $V_\psi$ makes training sensitive to mismatch between the learned policy and the dataset's behavior policy: on `Pendulum` with a TD3-generated dataset, SAC with an explicit $V$-network stabilizes quickly around an average reward of $-150$, whereas SAC without a $V$-network fluctuates heavily and never exceeds $-200$. In DR-SAC, the target network $V_{\bar{\psi}}(\tilde{s}_i')$ also scores all $m$ VAE-drawn next states directly inside the log-sum-exp without drawing and evaluating $m$ fresh actions through the critic ensemble. [Paper, Section 3.3 and Appendix C.3.3, Figure 9](https://arxiv.org/abs/2506.12622).
2. **Diversified $Q$-ensemble ($\text{SAC-}N$).** To curb offline overestimation bias, DR-SAC trains $n$ independent critics $\{Q_{\theta_i}\}_{i=1}^n$ ($n = 2$ on `Pendulum`, `Cartpole`, and `LunarLander`; $n = 5$ on `Reacher` and `HalfCheetah`) against the robust target $\widetilde{\mathcal{T}}_{\delta, \widetilde{g}^\star}^\pi Q_\theta(s,a)$ and takes $\min_{i \in [n]} Q_{\bar{\theta}_i}$ in the $V$-network and actor updates. [Paper, Section 3.3 and Appendix C.1, Table 2](https://arxiv.org/abs/2506.12622).

## 5. What robustness buys and what it costs

### 5.1 Robustness across five continuous-control environments

All methods are trained purely offline on $N = 10^6$ transitions collected in the nominal environment (via $\epsilon$-greedy rollouts around TD3 or SAC behavior policies, with $\epsilon = 0.5$ on `Pendulum`, `Cartpole`, and `LunarLander`, and $\epsilon = 0.3$ on `Reacher` and `HalfCheetah`) and evaluated over 50 episodes across parameter, actuator, and observation shifts. [Paper, Section 4.1 and Appendix C.1, Table 3](https://arxiv.org/abs/2506.12622).

| Environment | Ambiguity Radius | Perturbation Axis | Headline Empirical Result |
| --- | ---: | --- | --- |
| `LunarLander` | $\delta = 0.25$ | Joint main & side `engine_power`; linear `wind_power` | At $20\%$ engine power shift, DR-SAC averages **$240$** while SAC, CQL, FQI, and RFQI all drop below **$180$**; at $30\%$ engine power reduction, DR-SAC achieves **$9.8\times$** the average reward of SAC. |
| `Pendulum` | $\delta = 0.50$ | Pole `length`, `mass`, `gravity`, and heavy-tailed Cauchy state noise | **$35\%$** higher average reward than SAC at $20\%$ length shift; **$>40\%$** higher at $120\%$ mass shift; top curve across $40\%$ gravity shift and Cauchy noise scales. |
| `Cartpole` (continuous) | $\delta = 0.75$ | Random actuator probability and Gaussian observation noise | All models calibrated to $500$ nominal return; DR-SAC beats RFQI across random-action rates ($<50\%$) and improves return by **$>75\%$** over SAC and DDPG at observation noise $\sigma \in \{0.2, 0.3\}$. |
| `Reacher` (MuJoCo) | $\delta = 0.20$ | Gaussian state noise (dims 4–9) and `joint_damping` (2.0–7.0) | Highest mean return across all observation noise and joint damping levels, with lower or comparable variance (Table 4). |
| `HalfCheetah` (MuJoCo) | $\delta = 0.20$ | `back_damping`, `front_damping`, and `front_stiff` | Under up to $50\%$ `back_damping` shift, DR-SAC holds return **above $6{,}300$** while SAC falls **below $5{,}950$**; up to **$10\%$** gain at $80\%$ `front_stiff` shift. |

[Paper, Section 4.2, Figure 1, and Appendix C.2, Figures 4–6 and Table 4](https://arxiv.org/abs/2506.12622).

<figure>
  <img src="/figures/distributionally-robust/drsac-robust.png" alt="Six perturbation plots from the DR-SAC paper comparing DR-SAC against SAC, FQI, RFQI, DDPG, and CQL on Pendulum length, Cartpole action noise, LunarLander engine power, Reacher observation noise, Reacher joint damping, and HalfCheetah back damping.">
  <figcaption>Mean episode return over 50 evaluation rollouts (±0.5 standard deviation shaded) as deployment dynamics move away from the nominal offline training environment. [Paper, Figure 1].</figcaption>
</figure>

### 5.2 Compute accounting: functional vs. exact dual, and DR-SAC vs. RFQI

The paper's ablation tables quantify exactly where the computational time goes on identical hardware (Ubuntu 24.04 LTS, Intel Core i7-6850K CPU with 12 cores, 64 GB RAM, and one NVIDIA GTX 1080 Ti 11 GB GPU). [Paper, Section 4.3.1 and Appendices C.1 and C.3.1](https://arxiv.org/abs/2506.12622).

<figure>
  <img src="/blog/dr-sac/training-efficiency.svg" alt="Two-panel bar chart of training time and inner gradient descent steps from Tables 1, 5, and 6 of the DR-SAC paper. Panel A compares SAC, DR-SAC, FQI, and RFQI wall times on Cartpole, LunarLander, and Reacher. Panel B compares DR-SAC at 5 GD steps against RFQI at 5, 10, 100, and 1000 GD steps on LunarLander, plus the Pendulum accurate vs functional ablation.">
  <figcaption>Wall-clock training times and inner dual optimization steps from Tables 1, 5, and 6 of the paper.</figcaption>
</figure>

First, replacing per-$(s,a)$ scalar SciPy solves (`DR-SAC-Accurate`) with the shared functional network $\mathcal{G}_\eta$ (`DR-SAC-Functional`, both with $\delta = 0.5$) on `Pendulum` cuts training time by more than **98%** without losing robustness:

| `Pendulum` Variant | Training Steps | Wall Time (min) | Relative to `SAC` | Relative to `Accurate` |
| --- | ---: | ---: | ---: | ---: |
| `SAC` (non-robust baseline) | 10,000 | **1.7** | $1.0\times$ | $0.65\%$ |
| `DR-SAC-Functional` (Algorithm 1) | 10,000 | **4.7** | $2.8\times$ | **$1.81\%$** |
| `DR-SAC-Accurate` (per-$(s,a)$ SciPy) | 8,000 | **260.0** | $152.9\times$ | $100.0\%$ |

Even with 25% more gradient steps, `DR-SAC-Functional` finishes in under 2% of `DR-SAC-Accurate`'s wall time, matches its return across mass perturbations up to $120\%$, and outperforms it on length perturbations below $30\%$. [Paper, Section 4.3.1, Figure 2, and Appendix C.3.1, Table 5](https://arxiv.org/abs/2506.12622).

Second, across the three environments where RFQI converges, DR-SAC requires at most **$2.6\times$** the training time of non-robust SAC, whereas RFQI requires **11.3–14.0×** the time of FQI and up to **$23.2\times$** the time of DR-SAC (saving over $80\%$ of training time in every case):

| Environment | `SAC` (min) | `DR-SAC` (min) | `FQI` (min) | `RFQI` (min) | `RFQI` / `DR-SAC` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Cartpole` | 2 | **4** | 7 | 93 | **$23.25\times$** |
| `LunarLander` | 16 | **36** | 17 | 238 | **$6.61\times$** |
| `Reacher` | 13 | **32** | 14 | 159 | **$4.97\times$** |

Why is RFQI an order of magnitude slower when both algorithms use a functional dual network? Table 6 of the paper isolates the reason: **inner-loop conditioning**. In each critic update, DR-SAC's smooth KL log-exponential dual needs only **5 gradient descent (GD) steps** on $\mathcal{G}_\eta$, whereas RFQI's piecewise-linear TV dual on top of BCQ defaults to **1,000 GD steps**. Truncating RFQI's inner loop on `LunarLander` causes immediate degradation even in the nominal environment:

| Algorithm on `LunarLander` | Inner GD Steps / Update | Training Time (min) | Nominal Unperturbed Return |
| --- | ---: | ---: | ---: |
| **DR-SAC** | **5** | **36** | **240.0** |
| RFQI | 5 | 12 | 175.9 |
| RFQI | 10 | 21 | 181.9 |
| RFQI | 100 | 139 | 192.9 |
| RFQI (default) | 1,000 | 238 | 201.2 |

[Paper, Section 4.3.1, Table 1, and Appendix C.3.1, Figure 7 and Table 6](https://arxiv.org/abs/2506.12622).

### 5.3 Generative model overhead, theoretical regret, and conservatism limits

Replacing the conditional VAE with alternative generative families confirms that **sampling speed inside the Bellman backup** is the governing systems constraint:

- **Normalizing flows** exhibit unstable training even in unperturbed `Pendulum`.
- **Diffusion probabilistic models** match the VAE's robustness curves on `Pendulum` and `Cartpole`, but iterative denoising during next-state sampling makes training **at least $4.5\times$ slower** than VAE-based DR-SAC.
- **VAE latent dimension** $d_z$ is benign across a wide plateau: on `Pendulum` ($\dim(\mathcal{S})=3, \dim(\mathcal{A})=1$), varying $d_z \in \{5, 10, 20\}$ produces virtually indistinguishable robustness curves (while extreme under- or over-parameterization at $d_z = 1$ or $50$ degrades mildly). [Paper, Section 4.3.2, Figure 3, and Appendix C.3.2, Figure 8](https://arxiv.org/abs/2506.12622).

On the theoretical side, Appendix D quantifies how nominal transition estimation error propagates into the learned robust policy. Suppose the estimated nominal kernel $\hat{p}_{s,a}^0$ satisfies $D_{\mathrm{KL}}(p_{s,a}^0 \,\Vert\, \hat{p}_{s,a}^0) \le \varepsilon_1^2$ and shares support with the true nominal kernel, $\operatorname{supp}(p_{s,a}^0) = \operatorname{supp}(\hat{p}_{s,a}^0)$ (Assumption D.2, which implies $\operatorname{TV}(p_{s,a}^0, \hat{p}_{s,a}^0) \le \frac{1}{2}\varepsilon_1$ by Pinsker's inequality). Then **Lemma D.3** bounds the single-step operator discrepancy by

$$
\left\Vert \hat{\mathcal{T}}_\delta^\pi Q - \mathcal{T}_\delta^\pi Q \right\Vert_\infty \le \gamma \varepsilon_1 \frac{V_{\max}}{\delta} \exp\!\left(\frac{V_{\max}}{\underline{\beta}}\right) =: \varepsilon_2,
\qquad
V_{\max} = \frac{R_{\max} + \alpha \log \vert\mathcal{A}\vert}{1-\gamma},
$$

where $\underline{\beta} = \min\{\beta^\star/2, \hat{\beta}^\star/2, 1/2\} > 0$ lower-bounds the active dual temperature. Because both $\mathcal{T}_\delta^\pi$ and $\hat{\mathcal{T}}_\delta^\pi$ are $\gamma$-contractions, $\Vert \hat{Q}_{\mathcal{M}_\delta}^\pi - Q_{\mathcal{M}_\delta}^\pi \Vert_\infty \le \varepsilon_2 / (1-\gamma)$, and the distributionally robust regret of the converged policy $\hat{\pi}_{\mathcal{M}_\delta}^\star$ satisfies

$$
R_{\mathcal{M}_\delta}\!\left(\hat{\pi}_{\mathcal{M}_\delta}^\star\right) := \left\Vert V_{\mathcal{M}_\delta}^\star - V_{\mathcal{M}_\delta}^{\hat{\pi}_{\mathcal{M}_\delta}^\star} \right\Vert_\infty \le \frac{2\varepsilon_2}{1-\gamma}.
$$

[Paper, Appendix D, Definition D.1, Assumption D.2, and Lemma D.3](https://arxiv.org/abs/2506.12622).

Both the regret bound and the KL formulation point to the two structural costs of this approach:

1. **Support restriction ($\operatorname{supp}(p_{s,a}) \subseteq \operatorname{supp}(\hat{p}_{s,a}^0)$):** A KL ball only reweights transitions where the nominal model assigns positive density. A Gaussian-decoder VAE helps by smoothing empirical transitions over continuous state space, but shifts that jump outside the learned generative support cannot be anticipated by KL reweighting alone.
2. **Conservatism and manual radius selection ($\delta$):** Larger $\delta$ forces the critic to weight low-value next states more heavily, trading peak nominal return for a flatter degradation curve under perturbation. In the paper's experiments, the best ambiguity radius depends on task geometry ($\delta = 0.75$ on `Cartpole`, $0.50$ on `Pendulum`, $0.25$ on `LunarLander`, and $0.20$ on `Reacher` and `HalfCheetah`) and is chosen by grid search rather than derived automatically from local sample density. [Paper, Appendix C.2](https://arxiv.org/abs/2506.12622).

## 6. Takeaways

1. **The bottleneck in continuous-control DR-RL is the Bellman backup, not the minimax objective.** KL strong duality (Proposition 3.3) removes the infinite-dimensional transition kernel search and replaces it with an exponentially tilted expectation under the nominal model parameterized by a single scalar temperature $\beta \ge 0$, while preserving $\gamma$-contraction and monotonic soft policy improvement (Propositions 3.4–3.5, Theorem 3.6).
2. **Decomposable functional optimization makes the dual backup batch-friendly.** By Rockafellar–Wets interchange on $L^1(\mathcal{S}\times\mathcal{A})$ (Proposition 3.7), replacing per-$(s,a)$ scalar solves with a shared dual network $\mathcal{G}_\eta$ cuts training time on `Pendulum` from 260 minutes (`DR-SAC-Accurate`) to 4.7 minutes (`DR-SAC-Functional`)—a $>50\times$ speedup with no loss of robustness.
3. **Under KL duality, single-sample offline ERM silently degenerates to non-robust RL.** Because $-\beta \log \mathbb{E}[\exp(-V/\beta)]$ is nonlinear in the transition distribution, a single offline next-state draw cancels $\log$ and $\exp$, forcing the optimal dual multiplier to zero. Drawing $m$ synthetic next states from a conditional transition VAE—and scoring them with an explicit SAC-v1 target value network $V_{\bar{\psi}}$—resolves double sampling without a simulator and runs at least $4.5\times$ faster than diffusion-based transition models.
4. **Smooth KL duals converge in 5 inner gradient steps where TV duals need 1,000.** Across continuous-control benchmarks, DR-SAC trains in **2.0–2.5×** the wall time of standard SAC and **5–23×** faster than RFQI, while achieving up to $9.8\times$ higher return than SAC under severe environment perturbations.
