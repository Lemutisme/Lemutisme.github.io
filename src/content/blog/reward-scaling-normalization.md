---
title: "Your Reward Shaping Might Disappear at Normalization"
description: "Under GRPO's within-group centering and standard-deviation normalization, any positive per-prompt reward multiplier cancels out of the advantage; changing which prompts receive learning pressure requires either multiplying after standardization, using a nonlinear multi-level reward transform, or moving the scale into KL regularization."
publishedAt: 2026-11-08
tags:
  - LLM Post-Training
  - Reinforcement Learning
  - Reward Shaping
  - GRPO
draft: false
---

**Paper-Reading Note on:** **Controlling Sharpening in LLM Post-Training** · **arXiv:2611.03146 (2026)**

Simon Matrenok (EPFL), Nicolas Menet (ETH Zürich), Nicolas Flammarion (EPFL), Andreas Krause (ETH Zürich)

[arXiv](https://arxiv.org/abs/2611.03146) · [PDF](https://arxiv.org/pdf/2611.03146)

*Attribution note: This is a paper-reading research note written for this site, **not Duo Zhou's own work**. All theoretical derivations, closed-form optimal policies, algorithmic constructions (`KAPPO`), and empirical measurements discussed below are due to **Simon Matrenok, Nicolas Menet, Nicolas Flammarion, and Andreas Krause** ([arXiv:2611.03146](https://arxiv.org/abs/2611.03146)).*

A common instinct in reinforcement learning with verifiable rewards (RLVR) is to control which problems the model works hardest on by rescaling the reward per prompt—upweighting hard prompts where the base model rarely succeeds and downweighting easy prompts that it already solves reliably. In **Controlling Sharpening in LLM Post-Training**, Matrenok, Menet, Flammarion, and Krause (2026) prove that at the level of the KL-regularized objective, optimizing $\mathrm{pass@k}$ on binary rewards is *exact* single-reward ($\mathrm{@1}$) optimization with a prompt-adaptive reward multiplier $a_k(x) = k(1 - p^*(x))^{k-1}$.

Yet when that exact reward multiplier is handed directly to Group Relative Policy Optimization (GRPO; Shao et al., 2024), **within-group centering and standard-deviation normalization erase it**. Because $a_k(x) > 0$ is shared across all completions sampled for prompt $x$, it factors out of both the centered numerator and the group standard-deviation denominator and cancels identically (ignoring the denominator stabilizer). To make a per-prompt scale actually redistribute gradient pressure across prompts in standardized GRPO, **the multiplier must be applied after standardization**, outside the normalization quotient. [Matrenok et al., 2026, Section 5.2 and Appendix G](https://arxiv.org/abs/2611.03146).

<figure>
  <img src="/blog/reward-scaling-normalization/scale-before-after-grpo.svg" alt="Three-stage diagram comparing reward scaling in GRPO: (1) multiplying binary rewards by a positive per-prompt scale a_k(x) before within-group standardization cancels in numerator and denominator; (2) multiplying the standardized advantage after group normalization preserves a_k(x) via positive homogeneity of the clipped surrogate; (3) for continuous rewards with k greater than 1, the nonlinear win-rate transform S_x^q warps relative spacing within the group and survives standardization.">
  <figcaption>How GRPO's within-group standardization interacts with KAPPO's adaptive reward transformations across binary pre-scaling, binary post-scaling, and continuous nonlinear win-rate shaping (Matrenok et al., 2026, Section 5.2 and Appendix G).</figcaption>
</figure>

## 1. What group mean-and-std does to a reward

For a prompt $x \sim \mathcal{D}$, GRPO samples a group of $n$ completions $y_1, \dots, y_n \sim \pi_{\mathrm{old}}(\cdot \mid x)$ and evaluates their scalar rewards $R_1, \dots, R_n$, where $R_i = R(y_i, x)$. Write the empirical group mean and group standard deviation as

$$
\bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j,
\qquad
s_R = \operatorname{std}_{j=1,\dots,n}(R_j).
$$

Omitting the numerical denominator stabilizer for a moment and considering any non-degenerate group ($s_R > 0$; constant-reward groups with $s_R = 0$ receive zero advantages), GRPO defines the baseline-centered, variance-standardized advantage of completion $y_i$ as

$$
\widehat{A}_i = \frac{R_i - \bar{R}}{s_R}.
$$

Now suppose we replace the raw reward $R_i$ with any prompt-dependent positive affine transformation before group normalization:

$$
\widetilde{R}_i = a(x) R_i + b(x),
\qquad
a(x) > 0, \quad b(x) \in \mathbb{R}.
$$

Because $a(x)$ and $b(x)$ depend on the prompt $x$—or on reference-policy statistics of $x$—but not on the index $i \in \{1, \dots, n\}$ within the group, linearity of the sample mean and positive homogeneity of the standard deviation give

$$
\overline{\widetilde{R}} = \frac{1}{n}\sum_{j=1}^{n} \left(a(x) R_j + b(x)\right) = a(x)\bar{R} + b(x),
\qquad
\operatorname{std}_{j=1,\dots,n}(\widetilde{R}_j) = \vert a(x)\vert \, s_R = a(x) s_R.
$$

Substituting $\widetilde{R}_i$ into the within-group standardized advantage yields immediate cancellation:

$$
\frac{\widetilde{R}_i - \overline{\widetilde{R}}}{\operatorname{std}_{j=1,\dots,n}(\widetilde{R}_j)} = \frac{a(x)(R_i - \bar{R})}{a(x) s_R} = \frac{R_i - \bar{R}}{s_R} = \widehat{A}_i.
$$

[Matrenok et al., 2026, Appendix G, Equation (139)](https://arxiv.org/abs/2611.03146).

Centering removes $b(x)$; dividing by the group standard deviation removes $a(x) > 0$. Within-group $z$-scoring is therefore invariant to the entire group of positive per-prompt affine reward transformations.

## 2. Why a positive per-group scale disappears, and why continuous nonlinear transforms need not

### 2.1 Binary rewards and the pass-at-k multiplier

To see why this invariance matters, consider how Matrenok et al. (2026) derive `@k` Adaptive Pressure for Policy Optimization (`KAPPO`). Standard KL-regularized post-training maximizes

$$
\mathcal{J}(\pi) := \mathbb{E}_{x \sim \mathcal{D}}\!\left[\mathbb{E}_{y \sim \pi(\cdot \mid x)}[R(y,x)] - \beta \, \mathrm{KL}\!\left(\pi(\cdot \mid x) \,\Vert\, \pi_{\mathrm{ref}}(\cdot \mid x)\right)\right],
$$

whose closed-form global maximizer is the exponential tilt $\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp(R(y,x)/\beta)$. Replacing expected single-trial reward with a multi-sample objective $V_k(\pi; x)$—either $\mathrm{pass@k}(\pi; x) = \mathbb{P}[\exists\, i \in \{1,\dots,k\} : R(y_i, x) = 1]$ for binary rewards $R(y,x) \in \{0, 1\}$, or $\mathrm{max@k}(\pi; x) = \mathbb{E}[\max_{i=1,\dots,k} R(y_i, x)]$ for real-valued rewards—defines the KL-regularized $\mathrm{@k}$ objective

$$
\mathcal{J}_k(\pi) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[V_k(\pi; x) - \beta \, \mathrm{KL}\!\left(\pi(\cdot \mid x) \,\Vert\, \pi_{\mathrm{ref}}(\cdot \mid x)\right)\right].
$$

[Matrenok et al., 2026, Section 3, Equations (1)–(3)](https://arxiv.org/abs/2611.03146).

For binary rewards $R(y,x) \in \{0,1\}$, $V_k(\pi; x) = 1 - (1 - p_\pi(x))^k$ is concave in the success probability $p_\pi(x) = \mathbb{P}_{y \sim \pi(\cdot \mid x)}[R(y,x) = 1]$, and its directional derivative with respect to $\pi$ has score $A_\pi(y,x) = k(1 - p_\pi(x))^{k-1} R(y,x)$ (Appendix B.1, Equations (44)–(49)). Applying their variational optimality lemma (Lemma A.1), **Proposition 4.1** proves that the unique optimal policy for $\mathcal{J}_k$ is

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{R(y,x)}{\beta_{\mathrm{eff}}(x)}\right),
\qquad
\beta_{\mathrm{eff}}(x) = \frac{\beta}{k\left(1 - p^*(x)\right)^{k-1}},
$$

with partition function $Z(x) = 1 - p_{\mathrm{ref}}(x) + p_{\mathrm{ref}}(x)\exp(1/\beta_{\mathrm{eff}}(x))$, where $p_{\mathrm{ref}}(x) = \mathbb{P}_{y \sim \pi_{\mathrm{ref}}(\cdot \mid x)}[R(y,x) = 1]$ and $p^*(x) = \mathbb{P}_{y \sim \pi^*(\cdot \mid x)}[R(y,x) = 1]$. Summing $\pi^*(y \mid x)$ over successful completions yields **Proposition 4.2**: for every interior reference success rate $0 < p_{\mathrm{ref}}(x) < 1$, $p^*(x)$ is the unique root in $(0,1)$ of the scalar fixed-point equation

$$
k\left(1 - p^*(x)\right)^{k-1} = \beta \log \frac{p^*(x)\left(1 - p_{\mathrm{ref}}(x)\right)}{p_{\mathrm{ref}}(x)\left(1 - p^*(x)\right)}.
$$

[Matrenok et al., 2026, Section 4.1, Propositions 4.1–4.2, Equations (4)–(6), and Appendix B](https://arxiv.org/abs/2611.03146).

Because $\pi^*(y \mid x)$ depends on $R(y,x)$ and $\beta_{\mathrm{eff}}(x)$ only through their ratio, **Corollary 4.3** and **Corollary 4.4** establish two exact objective-level equivalences with ordinary $\mathrm{@1}$ optimization:

1. **Prompt-adaptive regularization (Corollary 4.3):** Optimize $\mathcal{J}(\pi)$ in Equation (1) with raw reward $R(y,x)$ and prompt-dependent KL coefficient $\beta_{\mathrm{eff}}(x)$.
2. **Prompt-adaptive reward scaling (Corollary 4.4):** Optimize $\mathcal{J}(\pi)$ in Equation (1) with constant KL coefficient $\beta$ and rescaled binary reward

$$
\widetilde{R}_k(y,x) = a_k(x) R(y,x),
\qquad
a_k(x) := \frac{\beta}{\beta_{\mathrm{eff}}(x)} = k\left(1 - p^*(x)\right)^{k-1}.
$$

[Matrenok et al., 2026, Section 4.1, Equation (7), and Appendix G, Equation (138)](https://arxiv.org/abs/2611.03146).

Equation (7) explains *how* $\mathrm{pass@k}$ counteracts sharpening: easy prompts with high $p_{\mathrm{ref}}(x)$ have $p^*(x)$ close to $1$, so $(1 - p^*(x))^{k-1} \ll 1$ and $a_k(x)$ is small (equivalently, $\beta_{\mathrm{eff}}(x)$ is large), curbing probability concentration on already-solved tasks; difficult prompts with small $p_{\mathrm{ref}}(x)$ receive a much larger multiplier $a_k(x)$ up to $k$, directing optimization pressure toward rare successes.

However, if a practitioner implements Corollary 4.4 by feeding $\widetilde{R}_i = a_k(x) R_i$ into standardized GRPO, **the multiplier $a_k(x)$ vanishes by Equation (139)**. More generally, *any* reward shaping on a binary reward $R_i \in \{0, 1\}$ that preserves the ordering between success and failure is an affine map $\widetilde{R}_i = (\widetilde{R}(1,x) - \widetilde{R}(0,x)) R_i + \widetilde{R}(0,x)$ on the two points $\{0, 1\}$. Within a group of $n$ completions for prompt $x$, there are only two possible reward levels, so standardizing by the group mean and standard deviation maps every non-degenerate binary group with $m \in \{1, \dots, n-1\}$ successes to the exact same two advantage values—regardless of how the two binary outcomes were shifted or positively scaled for prompt $x$.

### 2.2 Why continuous nonlinear transformations survive standardization

The situation changes when rewards take more than two values within a group, such as the fraction of passed test cases $R(y,x) \in [0, 1]$ in code generation. For real-valued rewards under $\mathrm{max@k}$, **Proposition 4.5** and **Corollary 4.6** show that the KL-regularized $\mathrm{max@k}$ optimal policy is recovered by optimizing the $\mathrm{@1}$ objective $\mathcal{J}(\pi)$ with the nonlinear reward transformation

$$
\widetilde{R}(y,x) = S_x(R(y,x)),
\qquad
S_x(r) = k \int_{-\infty}^{r} F_{\pi^*}(u \mid x)^{k-1}\,du,
$$

where $F_{\pi^*}(r \mid x) = \mathbb{P}_{y' \sim \pi^*(\cdot \mid x)}[R(y', x) \le r]$ is the optimal reward CDF. Because reconstructing $F_{\pi^*}$ numerically for every prompt requires solving a coupled forward ODE/recursion over the full reference distribution (Appendix D, Equations (90)–(110)), Matrenok et al. (2026) also derive a closed-form solution (**Proposition 4.7**) when the reward is expressed as the reference quantile (win rate) $Q_x(y) = F_{\mathrm{ref}}(R(y,x) \mid x) = \mathbb{P}_{y' \sim \pi_{\mathrm{ref}}(\cdot \mid x)}[R(y', x) \le R(y,x)]$:

$$
\pi_q^*(y \mid x) = \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{S_x^{\mathrm{q}}(R(y,x))}{\beta}\right),
$$

where the win-rate reward transformation $S_x^{\mathrm{q}}(r) = T_{k,\beta}(F_{\mathrm{ref}}(r \mid x))$ is given explicitly by

$$
S_x^{\mathrm{q}}(r) = \beta \log\!\left(C + \frac{1}{\beta}\Psi^{-1}\!\left(F_{\mathrm{ref}}(r \mid x)\right)^k\right),
\qquad
\Psi(z) = \int_0^z \frac{du}{C + \frac{1}{\beta}u^k},
\qquad
\Psi(1) = 1.
$$

[Matrenok et al., 2026, Section 4.2, Propositions 4.5–4.7, Equations (8)–(12), and Appendix F, Equations (121)–(136)](https://arxiv.org/abs/2611.03146).

Why does feeding $S_x^{\mathrm{q}}(R_i)$ directly into GRPO's within-group standardization (Section 5.2, Equation (15)) work when binary scaling failed? Appendix G ("Continuous reward transformations", Equations (141)–(144)) and Appendix K.2 (Lemma K.3, Equation (158)) identify two reasons:

1. **Nonlinear convexity warps relative within-group spacing.** Writing $S_x^{\mathrm{q}}(r) = T_{k,\beta}(q)$ in terms of the reference quantile $q = F_{\mathrm{ref}}(r \mid x) \in [0, 1]$ and $G = \Psi^{-1}$, differentiating gives

$$
T'_{k,\beta}(q) = k G(q)^{k-1},
\qquad
T''_{k,\beta}(q) = k(k-1)G(q)^{k-2}\!\left(C + \frac{G(q)^k}{\beta}\right).
$$

For $k > 1$, $T_{k,\beta}$ is strictly convex and nonlinear on $[0, 1]$ (while $k$-Lipschitz since $0 \le T'_{k,\beta}(q) \le k$). Within-group centering and standardization eliminate only *affine* changes in reward; because $T_{k,\beta}$ stretches gaps near the upper tail ($q \to 1$) relative to gaps in the lower tail ($q \to 0$), the ratios $(S_x^{\mathrm{q}}(R_i) - \overline{S_x^{\mathrm{q}}}) / \operatorname{std}_j(S_x^{\mathrm{q}}(R_j))$ differ from the untransformed standardized advantages $(R_i - \bar{R}) / s_R$.
2. **At initialization, GRPO normalization is common across prompts.** Under a continuous reference reward distribution, $Q_x(Y) = F_{\mathrm{ref}}(R(Y,x) \mid x) \sim \operatorname{Unif}[0,1]$ when $Y \sim \pi_{\mathrm{ref}}(\cdot \mid x)$ for *every* prompt $x$ (Appendix F, Equation (123), and Appendix G, Equation (142)). At the start of training ($\pi \approx \pi_{\mathrm{ref}}$), the population mean and variance of the transformed reward,

$$
\mu_{k,\beta} = \mathbb{E}_{U \sim \operatorname{Unif}[0,1]}\!\left[T_{k,\beta}(U)\right],
\qquad
\sigma_{k,\beta}^2 = \operatorname{Var}_{U \sim \operatorname{Unif}[0,1]}\!\left[T_{k,\beta}(U)\right],
$$

are independent of $x$. Thus, at the population level near initialization, GRPO standardization applies the *same* affine transformation $S_x^{\mathrm{q}}(r) \mapsto (S_x^{\mathrm{q}}(r) - \mu_{k,\beta})/\sigma_{k,\beta}$ to every prompt, preserving KAPPO's cross-prompt difficulty redistribution up to a global constant shift and scale. [Matrenok et al., 2026, Appendix G, Equations (141)–(144)](https://arxiv.org/abs/2611.03146).

## 3. KAPPO's fix: multiply after standardization

Appendix G of Matrenok et al. (2026) diagnoses *why* the theoretical equivalence in Corollary 4.4 breaks down when plugged naively into standardized GRPO: **standardized GRPO does not estimate the policy gradient of the expected-reward objective $\mathcal{J}(\pi)$**. Dividing each prompt's centered rewards by its empirical standard deviation $s_R$ multiplies that prompt's update by $1/s_R$, a policy- and sample-dependent weight that is absent from $\nabla_\theta \mathcal{J}(\pi_\theta)$. That same $1/s_R$ factor cancels any per-prompt multiplier $a_k(x)$ placed inside the reward.

To retain GRPO's within-group variance normalization while restoring the cross-prompt pressure redistribution $a_k(x) = k(1 - p^*(x))^{k-1}$, KAPPO applies the prompt multiplier **after** standardization:

$$
\widehat{A}_i^k = a_k(x) \, \frac{R_i - \bar{R}}{s_R}.
$$

[Matrenok et al., 2026, Appendix G, Equation (140)](https://arxiv.org/abs/2611.03146).

Why does multiplying *after* standardization pass cleanly through the rest of GRPO? Look at the per-token clipped surrogate in the GRPO loss (Section 5.2, Equation (15)):

$$
\mathcal{L}_{\mathrm{@k\text{-}GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D},\, y_1,\dots,y_n \sim \pi_{\mathrm{old}}(\cdot \mid x)}\!\left[\frac{1}{n}\sum_{i=1}^{n} \frac{1}{\vert y_i\vert}\sum_{t} u_{i,t}^{\mathrm{clip}}(\theta)\right],
$$

$$
u_{i,t}^{\mathrm{clip}}(\theta) = \min\!\left(\rho_{i,t}(\theta)\,\widehat{A}_i^k,\; \operatorname{clip}\!\left(\rho_{i,t}(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\widehat{A}_i^k\right),
\qquad
\rho_{i,t}(\theta) = \frac{\pi_\theta(y_i^t \mid y_i^{<t}, x)}{\pi_{\mathrm{old}}(y_i^t \mid y_i^{<t}, x)}.
$$

Because $a_k(x) \ge 0$ is non-negative and independent of $\theta$, and $\min(c u, c v) = c \min(u, v)$ for every $c \ge 0$, the clipped surrogate $u_{i,t}^{\mathrm{clip}}(\theta)$ is **positively homogeneous** in $\widehat{A}_i^k$. Multiplying the standardized advantage by $a_k(x)$ therefore scales prompt $x$'s contribution to both the GRPO loss and its gradient $\nabla_\theta \mathcal{L}_{\mathrm{@k\text{-}GRPO}}(\theta)$ by exactly $a_k(x)$—even when GRPO is trained without an explicit online KL penalty (using $\beta$ only during one-time reference calibration to solve Equation (6) for $p^*(x)$).

Matrenok et al. (2026, Appendix G) also note two alternative designs that recover the theoretical interfaces directly without post-hoc rescaling:

| Adaptation Route | Advantage / Loss Modification | Why the Prompt Scale Survives |
| --- | --- | --- |
| **Post-standardization scaling** (Appendix G, Eq. 140) | $\widehat{A}_i^k = a_k(x)\,(R_i - \bar{R})/s_R$ in unregularized GRPO | Applied outside $s_R$; positive homogeneity of $u_{i,t}^{\mathrm{clip}}$ scales the prompt gradient by $a_k(x)$. |
| **Mean-only centering** (Appendix G) | Drop $s_R$ division; retain group centering $\widetilde{R}_i - \overline{\widetilde{R}} = a_k(x)(R_i - \bar{R})$ | Centering is linear, so pre-scaled rewards $\widetilde{R}_i = a_k(x)R_i$ retain $a_k(x)$ directly. |
| **Regularization-side adaptation** (Corollary 4.3, Eq. 5 & 13) | Keep raw rewards; set per-prompt KL weight $\beta_{\mathrm{eff}}(x) = \beta / a_k(x)$ in KL-regularized GRPO or `@k`-DPO | Moves difficulty adaptation out of the normalized reward and into the KL regularizer. |
| **Single-completion regression** (`@k`-$A^\star$PO, Section 5.2, Eq. 14) | Regress $\beta \log(\pi_\theta / \pi_{\mathrm{ref}})$ onto $a_k(x)R(y,x) - \beta \log Z(x)$ using $1$ online completion per prompt | No group standardization exists; matches multi-completion PKPO within $0.5$ percentage points on MATH Level 5 (Table 1) with $50\%$ ($k=2$) to $75\%$ ($k=4$) fewer online rollouts. |

## 4. Caveats: optimization trajectory, zero-success banks, and the epsilon stabilizer

### 4.1 Same optimum does not mean same training trajectory

Corollaries 4.3, 4.4, and 4.6 are **fixed-point equivalences**: they guarantee that two KL-regularized objectives share the unique global maximizer $\pi^*$ (Lemma A.1, Equation (22)). They do not guarantee that two algorithms—or even the same optimizer under reward scaling versus KL rescaling—follow identical parameter trajectories during finite-step gradient descent.

In standardized GRPO, this distinction is even sharper:

- **No exact objective-level equivalence under $s_R$ division:** Because dividing by $s_R$ alters the expected gradient away from $\nabla_\theta \mathcal{J}(\pi_\theta)$, and because standard RLVR GRPO omits the online KL penalty (Section 5.2), Matrenok et al. (2026, Appendix G) explicitly characterize Equation (140) as an *empirical extension* that injects their theoretically derived cross-prompt weighting into GRPO's existing update rule, rather than an exact optimizer of $\mathcal{J}_k(\pi)$.
- **Distribution drift during continuous `@k`-GRPO training:** For continuous win-rate rewards $S_x^{\mathrm{q}}(r)$, the prompt-independence of $(\mu_{k,\beta}, \sigma_{k,\beta}^2)$ in Equations (143)–(144) holds at initialization when $\pi_\theta \approx \pi_{\mathrm{ref}}$. As $\pi_\theta$ diverges from $\pi_{\mathrm{ref}}$ over training, the win-rate distribution evolves differently across prompts, so $\operatorname{std}_{j=1,\dots,n}[S_x^{\mathrm{q}}(R_j)]$ becomes prompt-dependent again—layering GRPO's own variance reweighting on top of KAPPO's nonlinear reward transformation (Appendix G).
- **What is tested empirically in the paper:** While Appendix G derives the binary post-standardization formula (Equation (140)), the paper's binary-reward math experiments on MATH and AIME 2026 (`OLMo-3-7B-Instruct-SFT` and `Qwen3-4B-Base`, Section 6.2, Figure 2) evaluate `@k`-$A^\star$PO ($k \in \{1, 2, 4\}$) and `@k`-DPO ($k \in \{1, 2\}$) against standard $k=1$ GRPO, rather than running binary `@k`-GRPO. The paper's empirical `@k`-GRPO experiments (Section 6.3, Figure 4 right) are conducted in the continuous-reward code-generation setting (`Qwen3-4B-Base` trained on LeetCodeDataset and evaluated on the hard subset of LiveCodeBench v6 for $k \in \{1, 2, 4\}$).

### 4.2 Zero successes in a finite calibration bank does not mean zero support

KAPPO estimates prompt difficulty once before training from a static bank of $M = 30$ reference completions per prompt ($\widehat{p}_{\mathrm{ref}}(x) = N(x)/M$, where $N(x) = \sum_{j=1}^M \mathbb{I}[R(y_j^{\mathrm{ref}}, x) = 1]$; Appendix J, Equation (145)). Across the $7{,}500$ MATH training prompts, generating all $225{,}000$ reference completions takes $\approx 0.5$ hours vs. $10\text{–}12$ hours for a single 1,000-step training run on 32 NVIDIA GH200 GPUs ($4.2\%\text{–}5.0\%$ of one run's wall time, or $<1\%$ when amortized over six runs; Equation (147)).

A critical subtlety separates the **population boundary** $p_{\mathrm{ref}}(x) = 0$ from the **empirical boundary** $N(x) = 0$ (Appendices B.3, J.2, and K.1):

- **Population zero support ($p_{\mathrm{ref}}(x) = 0$):** If the true reference policy assigns zero probability to every valid solution, then no completion in the reference support $\mathcal{S}_x = \{y : \pi_{\mathrm{ref}}(y \mid x) > 0\}$ has $R(y,x) = 1$. Any policy $\pi$ with finite KL divergence $\mathrm{KL}(\pi(\cdot \mid x) \,\Vert\, \pi_{\mathrm{ref}}(\cdot \mid x)) < \infty$ is supported on $\mathcal{S}_x$ and therefore also has $p_\pi(x) = 0$. The reward term in $\mathcal{J}_{k,x}(\pi)$ is identically zero, and the unique KL-regularized optimal policy is $\pi^*(\cdot \mid x) = \pi_{\mathrm{ref}}(\cdot \mid x)$ with $p^*(x) = 0$ (Appendix B.3, Equation (65)).
- **Empirical zero successes ($N(x) = 0$ out of $M = 30$):** In a finite bank of $M = 30$ draws, observing $N(x) = 0$ does *not* imply $p_{\mathrm{ref}}(x) = 0$. A hard but solvable prompt with true success rate $q = p_{\mathrm{ref}}(x) > 0$ produces zero successes in $M$ independent trials with probability

$$
\mathbb{P}\!\left(N(x) = 0\right) = (1 - q)^M,
$$

which for $M = 30$ equals **0.740** at $q = 0.01$ and **0.042** at $q = 0.1$ (Appendix K.1). Meanwhile, the exact fixed-point solution $p^* = g(q)$ of Equation (6) rises steeply near zero: for $\beta = 0.1$ and $k = 4$, $g(0) = 0$ whereas at the very first positive empirical grid point $q = 1/30$, $g(1/30) \approx 0.553$ (Appendix K.1, Figure 5).
- **Upward pressure bias on hard prompts:** When $N(x) = 0$, the unsmoothed calibration rule sets $\widehat{p}^*(x) = 0$ and assigns the **maximum possible reward multiplier** $\widehat{a}_k(x) = k(1 - 0)^{k-1} = k$ (Appendix J.2, Equation (148), and Algorithm 1). On prompts where $p_{\mathrm{ref}}(x)$ is small but nonzero, zero-success banks therefore create an upward finite-sample bias in $\widehat{a}_k(x)$, concentrating maximum pressure on hard prompts (at $k = 4, \beta = 0.1$, $\widehat{p}^*$ has standard deviation $0.123$ and bias $-0.028$ at $p_{\mathrm{ref}} = 0.1$, dropping to $0.026$ and $0.001$ at $p_{\mathrm{ref}} = 0.5$; Appendix K.1). At the opposite endpoint $N(x) = M$ (all successes), unsmoothed $p^* = 1$ yields $a_k = 0$ and $\beta_{\mathrm{eff}} = +\infty$; binary `@k`-$A^\star$PO sets the regression target to $0$, whereas `@k`-DPO adds a failure pseudo-count $\alpha = 0.5$ ($\widehat{q}_{\mathrm{D}} = N/(M + 0.5)$) and caps $\widehat{\beta}_{\mathrm{eff}}(x) = \min\{\beta / a_k(\widehat{q}_{\mathrm{D}}),\, 2\}$ (Appendix J.2, Equation (149), and Lemma K.2).

### 4.3 The epsilon stabilizer effect

Equation (139) omits the numerical stabilizer added to the group standard deviation in code. As documented in Appendix J.2 ("Degenerate online groups"), the implementation divides centered rewards by the group standard deviation plus $\varepsilon_{\mathrm{std}} = 10^{-6}$. What happens if a positive multiplier $a_k(x) > 0$ is applied *before* standardization when $\varepsilon_{\mathrm{std}} > 0$ is present?

For a group with raw empirical standard deviation $s_R$, pre-scaling $\widetilde{R}_i = a_k(x) R_i$ yields

$$
\widehat{A}_i^{\mathrm{pre}}(\varepsilon_{\mathrm{std}}) = \frac{a_k(x)(R_i - \bar{R})}{a_k(x) s_R + \varepsilon_{\mathrm{std}}}.
$$

Two regimes follow immediately:

1. **Degenerate groups ($s_R = 0$):** When all $n$ completions achieve the same reward ($R_1 = \dots = R_n = \bar{R}$), the numerator $R_i - \bar{R} = 0$ vanishes identically, so $\widehat{A}_i^{\mathrm{pre}}(\varepsilon_{\mathrm{std}}) = 0$ (and post-standardization scaling also gives $0$).
2. **Non-degenerate groups ($s_R > 0$):** Dividing numerator and denominator by $a_k(x) > 0$ gives

$$
\widehat{A}_i^{\mathrm{pre}}(\varepsilon_{\mathrm{std}}) = \frac{R_i - \bar{R}}{s_R + \varepsilon_{\mathrm{std}} / a_k(x)}.
$$

Instead of multiplying the advantage by $a_k(x)$, pre-standardization scaling only rescales the stabilizer from $\varepsilon_{\mathrm{std}}$ to $\varepsilon_{\mathrm{std}} / a_k(x)$. On any mixed binary group of size $n = 8$ (the GRPO group size in Appendix H), at least one completion succeeds and at least one fails, so $s_R \ge \sqrt{7}/8 \approx 0.331$ (under $1/n$ variance, or $\sqrt{1/8} \approx 0.354$ under $1/(n-1)$ sample variance)—more than five orders of magnitude larger than $\varepsilon_{\mathrm{std}} = 10^{-6}$. For hard and moderate prompts where $a_k(x) \in [0.1, k]$, the relative discrepancy between $\widehat{A}_i^{\mathrm{pre}}(\varepsilon_{\mathrm{std}})$ and the unscaled advantage $(R_i - \bar{R})/(s_R + \varepsilon_{\mathrm{std}})$ is on the order of $10^{-5}$ to $10^{-6}$. The stabilizer only makes pre-scaling noticeable in the extreme limit $a_k(x) \lesssim \varepsilon_{\mathrm{std}} / s_R \approx 3 \times 10^{-6}$ (when $p^*(x) \to 1$ on a nearly deterministic easy prompt), where $\varepsilon_{\mathrm{std}} / a_k(x)$ grows large and damps the advantage toward zero—still failing to provide the intended proportional scaling $a_k(x) \in (0, k]$ across interior prompts.

## 5. Takeaways: a practical check for practitioners

Whenever you add reward shaping, difficulty reweighting, or curriculum scaling to a group-normalized policy-gradient pipeline (such as GRPO, Reinforce++, or PPO with per-prompt advantage standardization), run a four-step audit:

1. **Check whether your transform is within-group affine.** Write down your shaped reward $\widetilde{R}(y_i, x)$ for a fixed prompt $x$ across completions $i = 1, \dots, n$. If $\widetilde{R}(y_i, x) = a(x) R(y_i, x) + b(x)$ with $a(x) > 0$—for example, multiplying verifiable correctness by an offline difficulty weight, inverse pass rate, or KAPPO's $a_k(x) = k(1 - p^*(x))^{k-1}$—then within-group centering kills $b(x)$ and dividing by the group standard deviation kills $a(x)$ (up to $\mathcal{O}(\varepsilon_{\mathrm{std}} / (a(x)s_R))$).
2. **Remember that every order-preserving transform of a binary reward is affine.** Because a binary reward $R_i \in \{0, 1\}$ takes only two values, any function $f_x : \{0, 1\} \to \mathbb{R}$ with $f_x(1) > f_x(0)$ satisfies $f_x(R_i) = (f_x(1) - f_x(0)) R_i + f_x(0)$. No reward-space shaping applied *before* within-group standardization can alter the standardized advantages of a binary reward.
3. **To reweight prompts in GRPO, multiply *after* standardization (or drop $s_R$).** Apply the per-prompt multiplier outside the normalized advantage, $\widehat{A}_i^k = a_k(x)(R_i - \bar{R})/s_R$ (Equation (140)); positive homogeneity of the PPO/GRPO clipped surrogate guarantees that the prompt's gradient scales by $a_k(x)$. Equivalently, remove standard-deviation division and retain mean-only centering $R_i - \bar{R}$, or move the prompt weight into the KL coefficient $\beta_{\mathrm{eff}}(x) = \beta / a_k(x)$ (Corollary 4.3).
4. **Use nonlinear transforms on multi-level rewards, or single-completion regression when groups are unnecessary.** When rewards have multiple levels (such as test-case pass rates in $[0, 1]$), a strictly convex nonlinear map like KAPPO's win-rate transform $S_x^{\mathrm{q}}(r)$ alters relative spacing within the group and survives standardization (Section 5.2, Equation (15)). And when multi-completion groups are only being used as a baseline mechanism, single-completion `@k`-$A^\star$PO bypasses group normalization entirely—matching multi-completion PKPO within $0.5$ percentage points on MATH Level 5 while reducing online rollout generation by $50\%$ at $k=2$ and $75\%$ at $k=4$ (Table 1 and Appendix L).
