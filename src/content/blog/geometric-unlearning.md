---
title: "Euclidean Orthogonality Is Not Retain Invariance"
description: "When forget and retain gradients conflict, a Euclidean-orthogonal projection still perturbs retained behavior once AdamW rescales coordinates by its diagonal preconditioner; Geometric-Disentanglement Unlearning projects inside the optimizer's metric so the executed step is first-order retain-invariant across existing unlearning objectives."
publishedAt: 2026-11-08
tags:
  - Machine Unlearning
  - Language Models
  - Optimization Geometry
  - Trustworthy AI
draft: false
---

**Geometric-Disentanglement Language Model Unlearning** · **EMNLP 2026 Main**

**Duo Zhou**, Yuji Zhang, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, ChengXiang Zhai, Heng Ji, Huan Zhang

[arXiv](https://arxiv.org/abs/2511.17100) · [PDF](https://arxiv.org/pdf/2511.17100) · [Code](https://github.com/Lemutisme/Geometric-Unlearning) · [Project page](/projects/geometric-unlearning/)

Language model unlearning asks a post-training update to erase the influence of a forget set $D_f$ while leaving a retain set $D_r$ and general capabilities intact. Most algorithms formulate this as a scalarized objective $\mathcal{L}_f(\theta) + \lambda \mathcal{L}_r(\theta)$, hoping that a single weight $\lambda$ can balance targeted erasure against capability preservation. When the forget and retain gradients share coordinates and point into opposing half-spaces, however, scalar weighting only slides along a one-dimensional conflict. A natural geometric fix from multi-task optimization is to project the forget gradient onto the Euclidean orthogonal complement of the retain gradient before updating. Yet in language model post-training, **Euclidean orthogonality does not imply retain invariance**: once an adaptive optimizer such as Adam or AdamW rescales coordinates by its diagonal preconditioner $D_t^{-1}$, a Euclidean-orthogonal vector is rotated back into the retain-sensitive subspace and still changes the retain loss to first order.

Geometric-Disentanglement Unlearning (GU) resolves this mismatch by **defining both the retain-invariance condition and the orthogonal projection inside the metric induced by the optimizer itself**. Rather than replacing the unlearning loss, GU leaves the training objective intact and routes the forget and retain gradients through metric-orthogonal projectors before the preconditioned parameter step is applied. [Paper, Sections 1 and 3](https://arxiv.org/abs/2511.17100).

<figure>
  <img src="/blog/geometric-unlearning/fig1-projection-geometry.svg" alt="Two-panel diagram comparing Euclidean projection and optimizer-metric projection in Geometric-Disentanglement Unlearning. Panel A shows the retain subspace T_r, the Euclidean unit ball, the anisotropic H-metric unit ball, the raw forget gradient g_f, its Euclidean projection (which leaks into retain loss under D_t^-1), and GU's H-orthogonal projection P_perp^(H) g_f. Panel B compares scalarized loss weighting, Euclidean projection, and optimizer-metric GU.">
  <figcaption>Figure 1. Why the projection must live in the optimizer's metric. A Euclidean-orthogonal forget direction satisfies g_perp^T g_r = 0 on the raw gradient, but coordinate-wise rescaling by D_t^-1 rotates the executed step back into the retain subspace T_r. GU projects inside the optimizer-induced metric H_t so the actual parameter step produces zero first-order change on the retain loss. Adapted from Figure 1 of the paper.</figcaption>
</figure>

## 1. Why a scalar weight only slides along a one-dimensional conflict

Let $\pi_\theta$ be a language model with parameters $\theta \in \mathbb{R}^p$ initialized from a reference model $\pi_{\mathrm{ref}}$ trained on a corpus $D = D_f \cup D_r$, where $D_f \subseteq D$ is the forget set to be scrubbed and $D_r = D \setminus D_f$ is the retain set whose behavior must be preserved. Standard gradient-based unlearning samples a paired mini-batch $(x_f, x_r) \sim D_f \times D_r$ at each step and minimizes a scalarized combination of a forgetting objective $\mathcal{L}_f(\theta)$ and a retention objective $\mathcal{L}_r(\theta)$:

$$
\mathcal{L}_{\mathrm{joint}}(\theta) = \mathcal{L}_f(\theta) + \lambda \, \mathcal{L}_r(\theta), \qquad \lambda \ge 0.
$$

Here $\mathcal{L}_f(\theta) = -\mathbb{E}_{x_f \sim D_f}[\ell_f(x_f; \theta)]$ encourages the model to suppress responses tied to $D_f$ (via token-level negative log-likelihood ascent, preference ratios against $\pi_{\mathrm{ref}}$ such as DPO, NPO, or SimNPO, or logit/label calibration such as CEU, UNDIAL, WGA, or SatImp), while $\mathcal{L}_r(\theta) = \mathbb{E}_{x_r \sim D_r}[\ell_r(x_r; \theta)]$ anchors behavior on $D_r$. [Paper, Section 2, Equations (1)–(2)](https://arxiv.org/abs/2511.17100).

Write the raw Euclidean gradients at step $t$ as $g_f = \nabla_\theta \mathcal{L}_f(\theta_t)$ and $g_r = \nabla_\theta \mathcal{L}_r(\theta_t)$. Both gradients act on the same parameter vector $\theta \in \mathbb{R}^p$. Whenever forgotten and retained examples rely on overlapping internal representations, $g_f$ and $g_r$ share active coordinates and frequently point into opposing half-spaces ($g_f^\top g_r < 0$). Under that conflict, the combined gradient $g_f + \lambda g_r$ is confined to the two-ray cone spanned by $g_f$ and $g_r$:

- **Small $\lambda$** follows $g_f$ and allows the forgetting update to climb $\mathcal{L}_r$, causing collateral damage to retained knowledge and downstream fluency.
- **Large $\lambda$** follows $g_r$ and suppresses movement along $g_f$, stalling erasure of the forget set.

Sweeping $\lambda$ only trades one failure mode for the other along a one-dimensional frontier because a scalar loss weight never separates the component of $g_f$ that damages $\mathcal{L}_r$ from the component of $g_f$ that can reduce $\mathcal{L}_f$ without disturbing $D_r$. Resolving that interference requires operating on the **direction of the parameter update** rather than the weight of the loss. [Project page](/projects/geometric-unlearning/); [Paper, Sections 1 and 3.1](https://arxiv.org/abs/2511.17100).

## 2. Why Euclidean orthogonality fails under adaptive optimizers

In multi-task and continual learning, the textbook way to prevent one gradient from harming another is Euclidean orthogonal projection: remove the component of $g_f$ parallel to $g_r$ (or to a subspace $T_r$ of per-sample retain gradients) so that the projected forget vector $g_\perp$ satisfies

$$
g_\perp^\top g_r = 0.
$$

If the optimizer updated parameters by plain stochastic gradient descent $\Delta\theta_t = -\eta g_\perp$, a first-order Taylor expansion of $\mathcal{L}_r$ around $\theta_t$ would give $\mathcal{L}_r(\theta_t + \Delta\theta_t) - \mathcal{L}_r(\theta_t) \approx -\eta g_r^\top g_\perp = 0$.

In language model post-training, however, the step applied to the parameters is not a scalar multiple of the raw gradient. Adaptive optimizers such as Adam and AdamW maintain a coordinate-wise second-moment accumulator $\hat{v}_t \in \mathbb{R}_+^p$ and rescale the update by a positive-definite diagonal preconditioner $D_t^{-1}$ before moving $\theta_t$:

$$
\Delta\theta_t = -\eta \, D_t^{-1} \tilde{g}_t.
$$

Suppose we feed a Euclidean-orthogonal forget gradient $g_\perp$ (satisfying $g_\perp^\top g_r = 0$) into the preconditioned step $\Delta\theta_t = -\eta D_t^{-1} g_\perp$. Expanding the retain loss to first order around $\theta_t$ reveals why Euclidean orthogonality breaks down:

$$
\mathcal{L}_r(\theta_t + \Delta\theta_t) - \mathcal{L}_r(\theta_t) \approx \nabla_\theta \mathcal{L}_r(\theta_t)^\top \Delta\theta_t = -\eta \, g_r^\top D_t^{-1} g_\perp \neq 0.
$$

Because $D_t^{-1}$ rescales each coordinate $i$ by a different factor $(\sqrt{\hat{v}_{t,i}} + \varepsilon)^{-1}$ reflecting historical gradient variance, multiplying $g_\perp$ by $D_t^{-1}$ shears the vector out of the Euclidean orthogonal complement of $g_r$. Enforcing $g_\perp^\top g_r = 0$ imposes orthogonality on a raw gradient vector that the model never actually takes. [Project page](/projects/geometric-unlearning/); [Paper, Section 3.1 and Appendix A.1](https://arxiv.org/abs/2511.17100).

## 3. Projecting in the optimizer metric: the loss stays, the step changes

### 3.1 First-order retain invariance as metric orthogonality

To align the invariance guarantee with the executed step, GU formulates "no side effects on retained knowledge" directly in the geometry induced by the optimizer at iteration $t$. Freeze a symmetric positive definite (SPD) metric $H_t \succ 0$ on $\mathbb{R}^p$ with inner product and norm

$$
\langle u, v \rangle_{H_t} := u^\top H_t v, \qquad \Vert v \Vert_{H_t} := \sqrt{\langle v, v \rangle_{H_t}}.
$$

For Adam-family optimizers with second-moment estimate $\hat{v}_t$, the paper defines the diagonal whitening operator $W_t$ and induced metric $H_t$ on each selected parameter tensor as

$$
W_t = \operatorname{diag}\!\left(\frac{1}{\sqrt{\hat{v}_t + \varepsilon}}\right), \qquad H_t = W_t^\top W_t = \operatorname{diag}\!\left(\frac{1}{\hat{v}_t + \varepsilon}\right).
$$

More generally, for any optimizer applying a linear preconditioner $P_t$ with step direction $d_t = -P_t g_t$, setting $H_t = P_t^\top P_t$ equates standard Euclidean inner products in whitened coordinates ($\tilde{u} = W_t u$, $\tilde{v} = W_t v$) with $H_t$-inner products in parameter space: $\langle u, v \rangle_{H_t} = \langle W_t u, W_t v \rangle_2$. [Paper, Section 3.1, Appendix A.1, Equation (8), and Appendix B](https://arxiv.org/abs/2511.17100).

Rather than protecting only a single averaged retain gradient, GU requires first-order retain invariance across **every finite subset of retain examples**. For each $x_r \in D_r$, let $g(x_r) = \nabla_\theta \ell_r(x_r; \theta_t) \in \mathbb{R}^p$ be its sample gradient, and define the retain-gradient subspace and its $H$-orthogonal complement (dropping the subscript $t$ on $H$ within a fixed step):

$$
T_r(\theta) := \operatorname{span}\!\left\{\, g(x_r) : x_r \in D_r \,\right\} \subseteq \mathbb{R}^p,
\qquad
T_r(\theta)^{\perp_H} := \left\{\, v \in \mathbb{R}^p : \langle v, g \rangle_H = 0 \;\; \forall g \in T_r(\theta) \,\right\}.
$$

For any finite sample $S = \{x_{r_1}, \dots, x_{r_m}\} \subset D_r$ with empirical retain loss $\mathcal{L}_r^S(\theta) = \sum_{i=1}^m \ell_r(x_{r_i}; \theta)$, the batch gradient $\nabla_\theta \mathcal{L}_r^S(\theta) = \sum_{i=1}^m g(x_{r_i})$ lies in $T_r(\theta)$. **Proposition 3.1** (proved in Appendix C.1 as Proposition C.1) establishes the foundational equivalence:

$$
\Delta\theta \in T_r(\theta)^{\perp_H}
\quad \Longleftrightarrow \quad
\Delta^{(1)} \mathcal{L}_r^S(\theta; \Delta\theta) := \left\langle \nabla_\theta \mathcal{L}_r^S(\theta), \Delta\theta \right\rangle_H = 0 \quad \forall \text{ finite } S \subset D_r.
$$

(Identifying $\Delta^{(1)} \mathcal{L}_r^S$ with the true directional derivative $D\mathcal{L}_r^S(\theta)[\Delta\theta] = \langle \nabla_\theta^H \mathcal{L}_r^S(\theta), \Delta\theta \rangle_H$ via the $H$-gradient $\nabla_\theta^H \mathcal{L} := H^{-1}\nabla_\theta \mathcal{L}$ yields the identical subspace condition because $\operatorname{span}\{g(x_r)\}$ and $\operatorname{span}\{H^{-1}g(x_r)\}$ share the same orthogonal complement once paired with $H$.) [Paper, Section 3.1, Equation (3), Proposition 3.1, and Appendix C.1](https://arxiv.org/abs/2511.17100).

### 3.2 Tangential entanglement and steepest feasible forgetting

Let $U = [u_1, \dots, u_k] \in \mathbb{R}^{p \times k}$ be a basis of retain gradient directions collected from a retain mini-batch $B_r \subset D_r$ so that $T_r = \operatorname{range}(U)$. The $H$-orthogonal projector onto $T_r$ and the complementary projector onto $T_r^{\perp_H}$ are

$$
P_{T_r}^{(H)} = U \left(U^\top H U\right)^{-1} U^\top H,
\qquad
P_\perp^{(H)} = I - P_{T_r}^{(H)}.
$$

When $U$ is $H$-orthonormalized via Gram–Schmidt in whitened coordinates ($U^\top H U = I_k$), the projector simplifies to $P_{T_r}^{(H)} = U U^\top H$. Every forget gradient $g_f$ then decomposes uniquely into a retain-tangential part and a retain-normal part:

$$
g_f = \underbrace{P_{T_r}^{(H)} g_f}_{\in\, T_r\text{ (tangential)}} + \underbrace{P_\perp^{(H)} g_f}_{\in\, T_r^{\perp_H}\text{ (normal)}}.
$$

Only the tangential component $P_{T_r}^{(H)} g_f$ interacts with $\mathcal{L}_r$ to first order. Its $H$-norm defines a concrete, per-step measure of **retain–forget gradient entanglement**:

$$
\operatorname{ent}_H(g_f) := \left\Vert P_{T_r}^{(H)} g_f \right\Vert_H,
$$

which vanishes if and only if $g_f \in T_r^{\perp_H}$. [Paper, Section 3.1, Equations (4)–(6)](https://arxiv.org/abs/2511.17100).

The retain-invariant subspace $T_r^{\perp_H}$ has dimension $p - k$ and contains infinitely many safe directions. Which direction in $T_r^{\perp_H}$ achieves the fastest forgetting progress under an $H$-norm trust-region budget? Define the feasible ball of retain-invariant unit steps

$$
\mathcal{C} := \left\{\, \Delta\theta \in \mathbb{R}^p : U^\top H \Delta\theta = 0,\; \Vert \Delta\theta \Vert_H \le 1 \,\right\} = \left\{\, v \in T_r^{\perp_H} : \Vert v \Vert_H \le 1 \,\right\}.
$$

Because every $\Delta\theta \in \mathcal{C}$ satisfies $\langle g_r, \Delta\theta \rangle_H = 0$ for $g_r = \nabla_\theta^H \mathcal{L}_r(\theta_t) \in T_r$ and $\langle P_{T_r}^{(H)} g_f, \Delta\theta \rangle_H = 0$, the linearized change of both the forget objective $\mathcal{L}_f$ and the joint objective $\mathcal{L}_{\mathrm{joint}} = \mathcal{L}_f + \alpha \mathcal{L}_r$ collapses to $\langle P_\perp^{(H)} g_f, \Delta\theta \rangle_H$. By Cauchy–Schwarz (**Lemma 3.2**, proved in Appendix C.2), the unique steepest feasible descent direction (whenever $P_\perp^{(H)} g_f \neq 0$) for both $\mathcal{L}_f$ and $\mathcal{L}_{\mathrm{joint}}$ is the normalized $H$-projected forget gradient:

$$
\Delta\theta_f^\star = \operatorname{argmin}_{\Delta\theta \in \mathcal{C}} \langle g_f, \Delta\theta \rangle_H = \operatorname{argmin}_{\Delta\theta \in \mathcal{C}} \langle g_f + \alpha g_r, \Delta\theta \rangle_H = -\frac{P_\perp^{(H)} g_f}{\left\Vert P_\perp^{(H)} g_f \right\Vert_H}.
$$

Crucially, projecting out $P_{T_r}^{(H)} g_f$ does not cancel unlearning: unless $g_f$ lies entirely inside $T_r$ ($P_\perp^{(H)} g_f = 0$, meaning any first-order improvement on $\mathcal{L}_f$ necessarily degrades $\mathcal{L}_r$), the step $\Delta\theta_{\mathrm{GU}} = -\eta P_\perp^{(H)} g_f$ remains a strict first-order descent direction for the forget loss, $\Delta^{(1)} \mathcal{L}_f(\theta; \Delta\theta_{\mathrm{GU}}) = -\eta \Vert P_\perp^{(H)} g_f \Vert_H^2 < 0$. [Paper, Section 3.1, Lemma 3.2, Appendix C.2, and Appendix D.3](https://arxiv.org/abs/2511.17100).

### 3.3 Split projected update and second-order drift bounds

Rather than projecting the summed gradient $g_f + \alpha g_r$ (which would annihilate the retain gradient $g_r \in T_r$), GU routes the forget and retain branches through complementary projectors:

$$
\Delta\theta = -\rho \left( \underbrace{P_\perp^{(H)} g_f}_{\text{retain-orthogonal forgetting}} + \beta \underbrace{P_{T_r}^{(H)} g_r}_{\text{retain-tangent repair}} \right), \qquad \rho > 0,\; \beta \ge 0.
$$

Section 3.2 and Appendix C.3–C.4 prove three properties of this split step:

1. **First-order safety and retain monotonicity (Proposition 3.3):** Since $\langle g_r, P_\perp^{(H)} g_f \rangle_H = 0$ and $P_{T_r}^{(H)} g_r = g_r$, the first-order change in retain loss is

$$
\Delta^{(1)} \mathcal{L}_r = \langle g_r, \Delta\theta \rangle_H = -\rho \beta \Vert g_r \Vert_H^2 \le 0.
$$

   For $\beta = 0$, the update is strictly first-order neutral to $\mathcal{L}_r$; for $\beta > 0$ and $g_r \neq 0$, it strictly decreases $\mathcal{L}_r$ to first order.
2. **Exact one-step retain descent under $H$-smoothness (Corollary 3.4):** If $\nabla_\theta^H \mathcal{L}_r$ is $L_r^{(H)}$-Lipschitz under $\Vert \cdot \Vert_H$, then by $H$-orthogonality $\Vert \Delta\theta \Vert_H^2 = \rho^2 (\Vert P_\perp^{(H)} g_f \Vert_H^2 + \beta^2 \Vert g_r \Vert_H^2)$, yielding

$$
\mathcal{L}_r(\theta + \Delta\theta) \le \mathcal{L}_r(\theta) - \rho \beta \Vert g_r \Vert_H^2 + \frac{L_r^{(H)}}{2} \rho^2 \left( \left\Vert P_\perp^{(H)} g_f \right\Vert_H^2 + \beta^2 \Vert g_r \Vert_H^2 \right).
$$

   Thus $\mathcal{L}_r(\theta + \Delta\theta) < \mathcal{L}_r(\theta)$ holds strictly for any step size $0 < \rho < \frac{2\beta \Vert g_r \Vert_H^2}{L_r^{(H)}(\Vert P_\perp^{(H)} g_f \Vert_H^2 + \beta^2 \Vert g_r \Vert_H^2)}$, while at $\beta = 0$ retain drift is purely second-order, $\mathcal{L}_r(\theta + \Delta\theta) \le \mathcal{L}_r(\theta) + \frac{L_r^{(H)}}{2}\rho^2 \Vert P_\perp^{(H)} g_f \Vert_H^2 = \mathcal{L}_r(\theta) + O(\rho^2)$.
3. **One-step joint objective decomposition (Proposition 3.5 and Corollary C.1):** The first-order change of $\mathcal{L}_{\mathrm{joint}} = \mathcal{L}_f + \alpha \mathcal{L}_r$ is

$$
\Delta^{(1)} \mathcal{L}_{\mathrm{joint}} = \langle g_f + \alpha g_r, \Delta\theta \rangle_H = -\rho \left( \left\Vert P_\perp^{(H)} g_f \right\Vert_H^2 + \alpha \beta \Vert g_r \Vert_H^2 + \beta \left\langle P_{T_r}^{(H)} g_f, g_r \right\rangle_H \right),
$$

   which satisfies $\Delta^{(1)} \mathcal{L}_{\mathrm{joint}} \le -\rho \Vert P_\perp^{(H)} g_f \Vert_H^2 \le 0$ whenever $\beta = 0$ or $\alpha \Vert g_r \Vert_H \ge \Vert P_{T_r}^{(H)} g_f \Vert_H = \operatorname{ent}_H(g_f)$. [Paper, Section 3.2, Equations (7)–(11), and Appendices C.3–C.4](https://arxiv.org/abs/2511.17100).

## 4. A plug-in routing layer for GradDiff, NPO, SimNPO, and beyond

Because GU modifies how gradients are combined rather than how the forget loss is defined, it attaches directly to any gradient-based unlearning objective—including `GradDiff`, `CEU`, `DPO`, `NPO`, `SatImp`, `SimNPO`, `UNDIAL`, and `WGA`—without altering their loss functions. [Paper, Section 2, Section 3.1, and Appendix B](https://arxiv.org/abs/2511.17100).

### 4.1 Whitening, KL retain anchor, and sign-aware trust region

Algorithm 1 in Appendix B implements the optimizer-metric projection in five lightweight operations right before `optimizer.step()`:

1. **KL retain anchor:** To probe retained behavior with low-variance gradients and a zero-gradient baseline at $\pi_\theta = \pi_{\mathrm{ref}}$, GU defines the retain subspace anchor using forward KL divergence to the frozen reference model on a small retain mini-batch $B_r \subset D_r$:

$$
\mathcal{L}_r^{\mathrm{KL}}(\theta) = \mathbb{E}_{x \in D_r}\!\left[D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid x) \,\Vert\, \pi_{\mathrm{ref}}(\cdot \mid x)\right)\right].
$$

2. **Gradient recovery without a second forward pass:** After a standard backward pass on $\mathcal{L}_{\mathrm{tot}} = \gamma \mathcal{L}_f + \alpha \mathcal{L}_r$ populates $g_{\mathrm{tot}} = \gamma g_f + \alpha g_r$ in `p.grad`, GU computes $g_r = \nabla_\theta \mathcal{L}_r^{\mathrm{KL}}$ and algebraically recovers the raw forget gradient $g_f = (g_{\mathrm{tot}} - \alpha g_r)/\gamma$.
3. **Whitening by Adam's second moment:** Reusing the optimizer's existing state $v_p$ with zero additional state allocation, both gradients are whitened via $\tilde{g}_{f,p} = W_p g_{f,p}$ and $\tilde{g}_{r,p} = W_p g_{r,p}$ where $W_p = \operatorname{diag}(1 / \sqrt{v_p + \varepsilon})$. On scheduled refresh steps (every $E$ steps), $\tilde{g}_{r,p}$ is orthogonalized in `fp32` via Gram–Schmidt against the current rank-$k$ basis $U_p = [u_{p,1}, \dots, u_{p,m}]$ ($m \le k$, stored in `fp16`), appending the normalized residual if its relative norm exceeds threshold.
4. **Sign-aware tangential gate and anisotropic trust region:** In whitened coordinates, let $a_i = \langle \tilde{g}_f, u_i \rangle$ and $b_i = \langle \tilde{g}_r, u_i \rangle$ be the projections of the forget and retain gradients along basis vector $u_i$. Tangential components with $a_i b_i \ge -\tau$ ($\tau \ge 0$, default $\tau = 0$) push in the same direction as $g_r$ (which after gradient descent would *increase* $\mathcal{L}_r$) and are discarded. Opposite-signed components with $a_i b_i < -\tau$ actively *decrease* $\mathcal{L}_r$ to first order; GU retains them under an anisotropic norm cap relative to the safe normal component:

$$
\tilde{g}_f^{\mathrm{sel}} = P_\perp^{(H)} \tilde{g}_f + \tilde{g}_f^{\mathrm{tan,keep}},
\qquad
\left\Vert \tilde{g}_f^{\mathrm{tan,keep}} \right\Vert_2 = \left\Vert \sum_{i :\, a_i b_i < -\tau} a_i u_i \right\Vert_2 \le \kappa \left\Vert P_\perp^{(H)} \tilde{g}_f \right\Vert_2,
$$

   with default $\kappa = 0.5$. Because every kept term satisfies $-a_i b_i > \tau \vert a_i\vert \vert b_i\vert \ge 0$, this sign-aware rule strengthens Proposition 3.3 to $\Delta^{(1)} \mathcal{L}_r \le -\rho\beta \Vert P_{T_r}^{(H)} g_r \Vert_H^2 - \rho\tau \sum_{i \in \mathcal{K}} \vert a_i\vert \vert b_i\vert \le 0$ while bounding total step energy by $\Vert \tilde{g}_f^{\mathrm{sel}} \Vert_2 \le (1 + \kappa) \Vert P_\perp^{(H)} \tilde{g}_f \Vert_2$ without a line search.
5. **De-whitening into the base optimizer:** Finally, GU maps the whitened update back via $W_p^{-1}$ to overwrite the parameter gradient:

$$
g_p^\star = \gamma \, W_p^{-1} \tilde{g}_{f,p}^{\mathrm{sel}} + \alpha \, W_p^{-1} P_{T_r}^{(H)}(p) \tilde{g}_{r,p}.
$$

   When AdamW subsequently applies its preconditioner $W_p$, the $W_p$ and $W_p^{-1}$ factors cancel on the projected components, so the actual parameter step lands in the intended $H_t$-orthogonal direction. [Paper, Appendix B, Algorithm 1, Equation (12), and Appendix C.4](https://arxiv.org/abs/2511.17100).

### 4.2 Why low-rank streaming on top blocks avoids quadratic scaling

Prior projection-based unlearning methods built for image classifiers or small models (such as UNSC, PGU, SEMU, FG-OrIU, and OrthoGrad; reviewed in Appendix A.3) either form $d \times d$ activation covariance / Gram matrices across layers or run repeated layerwise SVDs and per-sample QR factorizations. At LLM hidden dimensions ($d \approx 4{,}096\text{–}8{,}192$), maintaining and factorizing $d \times d$ matrices across layers is prohibitive in both memory and wall-clock time.

GU avoids $O(d^2)$ matrices entirely. It restricts projection to an automatically detected subset of forget-sensitive tensors—the last $K$ Transformer blocks, final normalization layer(s), and output head—and maintains the retain basis $U_p \in \mathbb{R}^{p_{\mathrm{tensor}} \times k}$ via streaming vector operations in $O(k \, p_{\mathrm{tensor}})$ time per step. Profiling `SimNPO + GU` on `Llama-3.2-1B` on an NVIDIA H100-96GB GPU (Appendix E.4, Table 8) confirms that the overhead is negligible across ranks $k \in \{1, 8, 32\}$:

| Configuration (Llama-3.2-1B, H100) | Basis Rank | Refresh Period | Wall Time (s) | Time Overhead | Peak Memory (MB) | Forget ES (lower better) | Model Utility (higher better) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SimNPO` (baseline) | — | — | 226 | +0.0% | 20,291 | 0.28 | 0.59 |
| `SimNPO + GU` | $k = 1$ | 2 steps | 229 | **+1.3%** | 20,329 (+0.2%) | **0.12** | 0.59 |
| `SimNPO + GU` | $k = 8$ | 2 steps | 231 | **+2.2%** | 21,013 (+3.6%) | **0.12** | **0.60** |
| `SimNPO + GU` | $k = 32$ | 2 steps | 233 | **+3.0%** | 21,231 (+4.6%) | **0.12** | **0.60** |

End-to-end runtime increases by only **1.3–3.0%** and peak GPU memory by **under 5%**, while forget-set Extraction Strength drops by more than half ($0.28 \to 0.12$) even at $k = 1$, with $k = 8$ capturing the full utility gain. [Paper, Appendix A.3 and Appendix E.4, Table 8](https://arxiv.org/abs/2511.17100).

## 5. What the 72-configuration sweep and ablations show

### 5.1 Aggregate behavior across 72 TOFU configurations

The main evaluation on **TOFU** (200 fictitious author profiles with 20 QA pairs each, from the Open Unlearning suite) spans **72 configurations**: 3 model backbones (`Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`) $\times$ 3 deletion splits (`Forget-1%`, `Forget-5%`, `Forget-10%`) $\times$ 8 base unlearning objectives (`GradDiff`, `CEU`, `DPO`, `NPO`, `SatImp`, `SimNPO`, `UNDIAL`, `WGA`). Every configuration is scored on four axes:

- **Retain Extraction Strength (`ES Re.`, $\uparrow$):** how faithfully the model preserves facts from the retain split $D_r$.
- **Forget Extraction Strength (`ES Un.`, $\downarrow$):** residual regurgitation on the forget split $D_f$ under constrained prompts (lower means stronger forgetting).
- **Privacy (`Priv.`, $\uparrow$):** Membership Inference Attack (MIA) closeness to a model trained from scratch on $D_r$ alone ($1.0$ for the un-finetuned vanilla model; $0.0$ for the fully fine-tuned model).
- **Model Utility (`MU`, $\uparrow$):** composite score combining generation probability, ROUGE, and Truth Ratio across retain and real-world factual sets.

<figure>
  <img src="/blog/geometric-unlearning/fig2-tofu-aggregate.svg" alt="Horizontal stacked bar chart over all 72 TOFU configurations from Tables 3 and 4 of the paper. ES Re. improves in 66 of 72 (91.7% non-degraded), MU improves in 62 of 72 (86.1% non-degraded), ES Un. improves in 38 and stays unchanged in 13 (51 of 72 or 70.8% non-degraded), and Privacy improves in 49 of 72 (68.1% non-degraded). 29 of 72 configurations are strictly Pareto-dominant and 0 of 72 degrade on all four metrics.">
  <figcaption>Figure 2. Aggregate effect of attaching GU across all 72 [model, forget-ratio, objective] configurations on TOFU. Exact counts from Tables 3 and 4 of the paper (Appendix D.1–D.2).</figcaption>
</figure>

Across the 72 configurations (Tables 1, 3, and 4 of the paper):

- **Retain quality (`ES Re.`) is improved or preserved in 66 / 72 configurations (91.7%)**—strictly improved in 66, unchanged in 0, and slightly lower in 6.
- **General model utility (`MU`) is improved or preserved in 62 / 72 configurations (86.1%)**—strictly improved in 62 and lower in 10.
- **Forgetting efficacy (`ES Un.`) is improved or preserved in 51 / 72 configurations (70.8%)**—strictly reduced in 38, unchanged at the floor in 13, and higher in 21.
- **Privacy (`Priv.`) is improved or preserved in 49 / 72 configurations (68.1%)**.
- **Joint multi-metric dominance:** GU is **strictly Pareto-dominant** (no metric worse and at least one strictly better across all four axes) in **29 / 72 configurations (40.3%)**, improves or preserves both `ES Re.` and `MU` simultaneously in **58 / 72 configurations (80.6%)**, and degrades all four metrics simultaneously in **0 / 72 configurations (0.0%)**. [Paper, Section 4.2, Table 1, and Appendix D, Tables 3–4](https://arxiv.org/abs/2511.17100).

### 5.2 Headline gains on SimNPO, collapse recovery on CEU, and cross-benchmark results

Attaching GU to **SimNPO** (the strongest preference-based baseline in Open Unlearning) produces large simultaneous gains in forgetting, retention, privacy, and utility across TOFU, **MUSE** (`Llama-2-7b-hf` on Books and News verbatim memorization and QA), and **WMDP-cyber** (`zephyr-7b-beta` hazardous capability removal vs. MMLU):

| Benchmark & Split | Backbone | Base Objective | Metric | Baseline | With GU | Relative / Absolute Shift |
| --- | --- | --- | --- | ---: | ---: | --- |
| `TOFU` (`Forget-1%`) | `Llama-3.1-8B` | `SimNPO` | `ES Un.` (forget, lower better) | 0.3101 | **0.1177** | **−62.0%** forgetting ES (with `ES Re.` $0.8256 \to 0.8284$, `Priv.` $0.6178 \to 0.7810$) |
| `TOFU` (`Forget-5%`) | `Llama-3.2-3B` | `SimNPO` | `ES Re.` (retain, higher better) | 0.6068 | **0.7977** | **+31.5%** retain ES, **−55.9%** `ES Un.` ($0.2470 \to 0.1089$), **+59.8%** `Priv.` ($0.4046 \to 0.6467$) |
| `TOFU` (`Forget-10%`) | `Llama-3.1-8B` | `SimNPO` | `MU` (utility, higher better) | 0.6029 | **0.6519** | **+8.1%** model utility, **−46.0%** `ES Un.` ($0.2110 \to 0.1140$), **+46.7%** `Priv.` ($0.4789 \to 0.7027$) |
| `TOFU` (`Forget-10%`) | `Llama-3.2-3B` | `UNDIAL` | `ES Re.` (retain, higher better) | 0.3538 | **0.7869** | **+122.4%** retain ES ($2.22\times$), `MU` $0.6550 \to 0.6992$, `ES Un.` $0.0416 \to 0.0396$ |
| `TOFU` (`Forget-10%`) | `Llama-3.1-8B` | `CEU` | `MU` (utility, higher better) | 0.0000 | **0.6773** | Recovers from complete utility collapse ($\mathrm{MU}=0$) while holding `ES Un.` at $0.0325$ |
| `MUSE-News` | `Llama-2-7b-hf` | `SimNPO` | `ES Un.` (forget, lower better) | 0.1778 | **0.0957** | **−46.2%** forgetting ES (`ROUGE Re.` $0.4114 \to 0.4143$) |
| `MUSE-Books` | `Llama-2-7b-hf` | `SimNPO` | `ROUGE Re.` & `Priv. Leak.` | 0.5103 / −54.25 | **0.5980 / −46.49** | **+17.2%** retained ROUGE, **−14.3%** privacy-leakage magnitude, **−42.2%** `ES Un.` ($0.1407 \to 0.0813$) |
| `WMDP-cyber` | `zephyr-7b-beta` | `SimNPO` / `WGA` | `Un. acc.` / `MMLU` | 0.4192 / 0.2550 | **0.4177 / 0.5498** | `SimNPO` lowers hazardous accuracy by **0.36%** ($0.4192 \to 0.4177$) with `MMLU` $0.5658 \to 0.5663$; `WGA` restores `MMLU` from $0.2550 \to 0.5498$ |

[Paper, Abstract, Section 1, Section 4.2, Tables 1–2, and Appendix D.4–D.5](https://arxiv.org/abs/2511.17100).

Two structural patterns stand out in Table 1 and Table 2:

1. **Benefits grow with model scale and deletion ratio.** As the forget set expands from `Forget-1%` to `Forget-10%` or the backbone scales from `1B` to `8B`, forget and retain gradients become more entangled. Unconstrained objectives such as `CEU` collapse completely to $\mathrm{MU} = 0.0000$ at `Forget-5%` and `Forget-10%` across all three Llama-3 scales; projecting out the retain-tangential step restores `MU` to $0.5635\text{–}0.6773$ while keeping `ES Un.` at the $0.0325$ floor.
2. **WMDP-cyber is primarily a capability-preservation result.** On `WMDP-cyber`, GU lowers hazardous accuracy in 4 of 7 objectives (`GD` $0.2420 \to 0.2375$, `SatImp` $0.4177 \to 0.4157$, `SimNPO` $0.4192 \to 0.4177$, `UNDIAL` $0.3829 \to 0.3789$) while maintaining or increasing general `MMLU` across all 7 families. Because the projection pulls updates toward the retain manifold, its primary effect on `WMDP` is preventing general-capability collapse (such as lifting `WGA` MMLU from $0.2550$ to $0.5498$) rather than acting as a standalone capability-erasure loss. [Paper, Appendix D.5](https://arxiv.org/abs/2511.17100).

### 5.3 Direct ablation of Euclidean versus Adam-diagonal geometry

Does the optimizer metric $H = W_t^\top W_t$ actually improve over Euclidean projection ($H = I$) when everything else in GU is held fixed? Appendix E.1 (Table 5) runs this exact control on `Llama-3.2-1B` and `Llama-3.2-3B` with `SimNPO + GU`:

<figure>
  <img src="/blog/geometric-unlearning/fig3-metric-ablation.svg" alt="Two-panel grouped bar chart of Table 5 from the paper comparing Euclidean (H = I) and Adam-diagonal (H = W_t^T W_t) projection metrics for SimNPO + GU on Llama-3.2-1B and Llama-3.2-3B across ES Re., ES Un., Privacy, and Model Utility.">
  <figcaption>Figure 3. Metric geometry ablation on TOFU (SimNPO + GU) from Table 5 of the paper (Appendix E.1). Lower is better for ES Un.; higher is better for ES Re., Priv., and MU.</figcaption>
</figure>

At both model scales, Euclidean projection ($H = I$) already improves substantially over unprojected `SimNPO` (on `3B`, `ES Un.` drops from $0.39$ to $0.21$ and `Priv.` rises from $0.63$ to $0.76$), confirming that removing retain-gradient conflict is the primary mechanism. Switching from $H = I$ to the Adam-diagonal metric $H = W_t^\top W_t$ then improves **all four metrics at both 1B and 3B**:

- On **`Llama-3.2-1B`**, `ES Re.` rises from $0.63 \to \mathbf{0.65}$, `ES Un.` drops from $0.26 \to \mathbf{0.21}$ ($-19.2\%$), `Priv.` rises from $0.70 \to \mathbf{0.73}$, and `MU` rises from $0.59 \to \mathbf{0.60}$.
- On **`Llama-3.2-3B`**, `ES Re.` rises from $0.76 \to \mathbf{0.78}$, `ES Un.` drops from $0.21 \to \mathbf{0.17}$ ($-19.0\%$), `Priv.` rises from $0.76 \to \mathbf{0.78}$, and `MU` rises from $0.64 \to \mathbf{0.65}$.

Appendix E.2–E.3 (Tables 6 and 7) further verify that GU's gains are neither an artifact of AdamW nor random-seed noise: under **SGD** on `Llama-3.2-3B` (`Forget-5%`), adding GU to `SimNPO` lifts `ES Re.` from $0.72 \to 0.74$, cuts `ES Un.` from $0.42 \to 0.20$, lifts `Priv.` from $0.61 \to 0.75$, and raises `MU` from $0.62 \to 0.63$; across random seeds on `Llama-3.2-3B`, `SimNPO + GU` exhibits standard deviations of $\pm 0.01$ on every metric against effect sizes of $-0.22$ on `ES Un.` and $+0.15$ on `Priv.`. [Paper, Appendix E.1–E.3, Tables 5–7](https://arxiv.org/abs/2511.17100).

## 6. Open questions: local first-order orthogonality versus long-horizon curvature

GU's theoretical guarantee is **local and first-order**. That scope makes the projection cheap enough to run online at LLM scale, and it also defines the boundary of what stepwise gradient projection can guarantee:

1. **Second-order curvature drift and rotating retain subspaces over long horizons:** Corollary 3.4 shows that even when the first-order term $\Delta^{(1)} \mathcal{L}_r$ is zero at each step, a step of size $\rho$ incurs second-order retain drift bounded by $\frac{L_r^{(H)}}{2} \rho^2 \Vert P_\perp^{(H)} g_f \Vert_H^2 = O(\rho^2)$. Over long post-training trajectories, these $O(\rho^2)$ curvature errors can accumulate as the local retain subspace $T_r(\theta_t)$ rotates along the optimization path. While GU's anisotropic trust region ($\kappa = 0.5$) and tangential repair ($\beta P_{T_r}^{(H)} g_r$) damp local drift, controlling cumulative displacement globally from initialization $\theta_0$ requires bounding the path rather than the single step—the question tackled from a global trajectory perspective in [Cumulative Fisher Protection](/projects/cumulative-fisher/).
2. **Dependence on a representative retain anchor and mini-batch subspace estimation:** Because $\widehat{T}_r(\theta_t)$ is estimated from a finite retain mini-batch $B_r \subset D_r$ at rank $k$ on selected top layers, any principal angle $\Theta(T_r, \widehat{T}_r)$ between the population retain subspace and its low-rank mini-batch estimate leaves a residual first-order leakage of order $O(\sin \Theta(T_r, \widehat{T}_r))$ (Appendix C.3 Remark), and settings without access to representative retain data cannot form $T_r$ directly.
3. **Diagonal metric approximation:** Binding $H_t$ to Adam's diagonal second-moment state captures coordinate-wise rescaling for free, but ignores off-diagonal cross-parameter curvature within and across attention and MLP blocks.

Together with [Cumulative Fisher Protection](/projects/cumulative-fisher/) and [Gradient Interference in Group-Aware RLHF](/projects/gradient-interference-rlhf/), GU points to a unified view of language model post-training: tradeoffs between competing behavioral specifications are governed by the **geometry of executed parameter updates** rather than scalar weights in the training loss. [Project page](/projects/geometric-unlearning/); [Paper, Section 6 and Appendix C.3](https://arxiv.org/abs/2511.17100).

## 7. Takeaways

1. **Scalar loss weights cannot disentangle shared parameters.** When forget and retain gradients point into opposing half-spaces ($g_f^\top g_r < 0$), sweeping $\lambda$ in $\mathcal{L}_f + \lambda \mathcal{L}_r$ only slides along a one-dimensional conflict between collateral damage and incomplete forgetting.
2. **Euclidean orthogonality is not retain invariance under adaptive optimizers.** A forget vector satisfying $g_\perp^\top g_r = 0$ still changes retain loss to first order by $-\eta g_r^\top D_t^{-1} g_\perp \neq 0$ once AdamW applies its coordinate-wise preconditioner $D_t^{-1}$. First-order retain invariance across all finite retain subsets holds if and only if the update lies in the $H_t$-orthogonal complement $T_r(\theta_t)^{\perp_{H_t}}$ induced by the optimizer's metric (Proposition 3.1).
3. **The loss stays; the step changes.** Within the retain-invariant trust region, the normalized $H_t$-projected forget gradient $-P_\perp^{(H_t)} g_f / \Vert P_\perp^{(H_t)} g_f \Vert_{H_t}$ is the unique steepest feasible descent direction for both $\mathcal{L}_f$ and $\mathcal{L}_{\mathrm{joint}}$ (Lemma 3.2), while the tangential norm $\operatorname{ent}_{H_t}(g_f) = \Vert P_{T_r}^{(H_t)} g_f \Vert_{H_t}$ measures per-step forget–retain entanglement.
4. **GU plugs into existing unlearning losses at <3% runtime and <5% memory overhead.** By whitening gradients with Adam's existing second-moment state and maintaining a low-rank ($k = 8$) Gram–Schmidt basis on top Transformer blocks and the output head, GU attaches to `GradDiff`, `NPO`, `SimNPO`, `UNDIAL`, `CEU`, `SatImp`, `DPO`, and `WGA` without modifying their objectives.
5. **Across 72 TOFU configurations, GU improves or preserves retain quality in 66/72 (91.7%) and model utility in 62/72 (86.1%), with up to 62% stronger forgetting on SimNPO.** Direct ablation confirms that projecting in the Adam-diagonal metric outperforms Euclidean projection ($H = I$) across all four evaluation axes on both `Llama-3.2-1B` and `Llama-3.2-3B`.
