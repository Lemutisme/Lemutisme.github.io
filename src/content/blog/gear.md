---
title: "GEAR: Scaling Global Optimization with Reusable Linear Bounds"
description: "GEAR makes global search scale in two ways: GPU batches reduce the effective cost of processing a domain, while reusable constraint-aware linear bounds close optimality proofs across 421 of 505 bounded continuous NLPs."
publishedAt: 2026-10-03
updatedAt: 2026-11-07
tags:
  - Global Optimization
  - Nonlinear Programming
  - Linear Bound Propagation
  - GPU Computing
draft: false
---

**GEAR: A GPU-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation** · **NeurIPS 2026**

**Duo Zhou**, Hesun Chen, Xiangru Zhong, Grani A. Hanasusanto, Huan Zhang

[arXiv](https://arxiv.org/abs/2611.10487) · [PDF](https://arxiv.org/pdf/2611.10487) · [OpenReview](https://openreview.net/forum?id=3WdBmyLLLC) · [Project page](/projects/gear/)

Global optimization becomes expensive when a solver must inspect thousands or millions of regions before it can certify a solution. Each region needs an objective lower bound, an infeasibility test, a tightened domain, a search for a feasible incumbent, and a decision about the next split. GEAR makes all five operations consume a single set of propagated linear bounds, allowing an active frontier of boxes to be processed as one GPU tensor workload and making the geometry from each propagation pass useful throughout the search ([Paper, Sections 1 and 3, Figure 1](https://arxiv.org/abs/2611.10487)).

There are two sources of scalability here: **lower the effective cost of processing a domain, and reduce the number of domains that need processing.** The interaction between them—and where it trades off against per-node polyhedral LP strength—explains the solver's behavior across problem classes far better than the hardware label alone.

## Account for the cost of a proof

Consider a bounded continuous nonlinear program over an initial finite box $B_0 \subset \mathbb{R}^n$:

$$
p^\star = \min_{x\in B_0} f_0(x)
\quad\text{subject to}\quad
f_i(x)\leq 0,\qquad i=1,\ldots,m.
$$

Source equalities and two-sided inequalities are written as pairs of one-sided inequalities with an absolute feasibility tolerance $\tau_{\mathrm{feas}} = 10^{-6}$ absorbed into each $f_i$ (for example, $g(x) = 0$ becomes $g(x) - \tau_{\mathrm{feas}} \leq 0$ and $-g(x) - \tau_{\mathrm{feas}} \leq 0$), so primal feasibility checks and dual lower bounds refer to the exact same problem ([Paper, Section 2, Equation (1), and Appendix A.3](https://arxiv.org/abs/2611.10487)). On any sub-box $B = [\underline{x}, \overline{x}] \subseteq B_0$, write the feasible set and local optimum as $\mathcal{F}(B) = \{x \in B : f_i(x) \leq 0,\; i = 1, \ldots, m\}$ and $p^\star(B) = \min_{x \in \mathcal{F}(B)} f_0(x)$ (Equation (3)). A run certifies global optimality when its verified feasible incumbent $P = f_0(\bar{x})$ and global lower bound $D \leq p^\star$ satisfy the relative gap criterion ([Paper, Section 4, Equation (13)](https://arxiv.org/abs/2611.10487)):

$$
\frac{P - D}{\max(1, \lvert P\rvert, \lvert D\rvert)} \leq 10^{-4}.
$$

Extending linear bound propagation (LBP) from neural network verification to constrained global optimization is not automatic ([Paper, Section 3.1](https://arxiv.org/abs/2611.10487)). A verifier tests a property against a fixed threshold and never needs a feasible point. In constrained NLP, the pruning threshold is the objective value of a feasible incumbent that the solver must find and improve, and an objective-only lower bound minimizes $f_0$ over the entire box—including infeasible points—so it can remain strictly below $p^\star$ no matter how finely the box is split. Minimizing $x$ on $[0, 1]$ subject to $x \geq 1/2$ has optimum $1/2$, yet every sub-box containing $0$ retains an objective-only lower bound of $0$; branching alone never prunes that box by dominance unless the constraint rules it out or raises its lower bound ([Paper, Section 3.1](https://arxiv.org/abs/2611.10487)).

For a spatial branch-and-bound run with batch size $K$, a useful runtime accounting model is

$$
T_{\mathrm{search}}\approx
N_{\mathrm{visited}}\frac{T_{\mathrm{batch}}}{K}
+T_{\mathrm{queue}}.
$$

The first factor measures how many boxes the tree must visit; the second measures the effective cost per box across bounding, clipping, and primal search. GEAR targets both factors simultaneously by turning propagated linear bounds into a shared constraint-aware representation on the GPU.

## Batch the nodes and the constraint outputs together

GEAR compiles the objective and all $m$ constraints as the outputs of a single computation graph $F(x) = (f_0(x), f_1(x), \ldots, f_m(x))$ ([Paper, Section 2, Equation (2), and Section 3.2](https://arxiv.org/abs/2611.10487)). Forward evaluation, automatic differentiation, and backward linear bound propagation operate on this same graph. Linear constraints $f_i(x) = a_i^\top x + c_i$ require no relaxation: LBP substitutes affine operators exactly, and when the parser exposes linear constraints as an explicit matrix, that matrix is reused directly across subproblems ([Paper, Section 3.2 and Appendix A.2](https://arxiv.org/abs/2611.10487)).

On each sub-box $B = [\underline{x}, \overline{x}]$, backward linear bound propagation substitutes valid local linear relaxations $\underline{h}(y) \leq h(y) \leq \overline{h}(y)$ from the outputs toward the inputs according to the sign of each backward weight ($\gamma \underline{h}(y)$ for $\gamma \geq 0$ and $\gamma \overline{h}(y)$ for $\gamma < 0$; Section 2, Equations (4)–(5), and Appendix A.2, Equation (15)), producing valid linear lower bounds on $\mathcal{F}(B)$:

$$
\underline{f}_i(x) = a_i^\top x + c_i \leq f_i(x) \qquad \forall x \in \mathcal{F}(B), \quad i = 0, \ldots, m.
$$

Across a batch of $K$ boxes in $n$ variables with $m$ constraints, the coefficients form dense tensors ([Paper, Section 3.2, Equation (7), and Appendix A.2, Equation (17)](https://arxiv.org/abs/2611.10487)):

$$
A \in \mathbb{R}^{K \times (m+1) \times n}, \qquad
c \in \mathbb{R}^{K \times (m+1)}.
$$

The batch dimension indexes heterogeneous boxes in the branch-and-bound frontier; the output dimension indexes the objective and constraints. Because every box shares the same graph topology while carrying its own interval and affine bounds, one batched GPU pass bounds thousands of subproblems without building a separate optimization model per node.

## Turn one propagation pass into five kinds of progress

A linear lower bound carries more information than its scalar box minimum $\min_{x \in B} \underline{f}_i(x) = c_i + \sum_{j=1}^n \min\{a_{i,j}\underline{x}_j, a_{i,j}\overline{x}_j\}$ (Section 2, Equation (6)). Its slope vector $a_i$ records how the function varies across $B$. Stacking the constraint linear bounds (together with any exact linear rows and presolve cuts) as $A = [a_1, \ldots, a_m]^\top$ and $c = (c_1, \ldots, c_m)^\top$ defines an outer polyhedral relaxation of the feasible set on $B$ ([Paper, Section 3.3, Equation (8)](https://arxiv.org/abs/2611.10487)):

$$
\mathcal{P}(B) = \{x \in B : Ax + c \leq 0\} \supseteq \mathcal{F}(B),
\qquad
p^{\mathrm{LP}}(B) = \min_{x \in \mathcal{P}(B)} \underline{f}_0(x) \leq p^\star(B).
$$

Instead of calling a simplex or interior-point LP solver at every box to evaluate $p^{\mathrm{LP}}(B)$, GEAR reuses $(a_0, c_0)$ and $(A, c)$ in five ways ([Paper, Figure 1 and Sections 3.3–3.5](https://arxiv.org/abs/2611.10487)):

1. **Lagrangian objective lower bound:** Dualizing the constraint rows $Ax + c \leq 0$ with nonnegative multipliers $\lambda \geq 0$ while keeping the box $B = [\underline{x}, \overline{x}]$ as the inner minimization domain yields the Lagrangian bound ([Paper, Section 3.3, Equation (9), and Appendix A.4, Proposition A.1](https://arxiv.org/abs/2611.10487)):

$$
d_B(\lambda) = \min_{x \in B}\left[\underline{f}_0(x) + \lambda^\top (Ax + c)\right]
= c_0 + c^\top \lambda + \sum_{j=1}^n \min\left\{a_j(\lambda)\underline{x}_j,\; a_j(\lambda)\overline{x}_j\right\},
$$

where $a(\lambda) = a_0 + A^\top \lambda$. At every feasible point $x \in \mathcal{F}(B)$, $\underline{f}_i(x) \leq f_i(x) \leq 0$, so $d_B(\lambda) \leq p^{\mathrm{LP}}(B) \leq p^\star(B)$ for any $\lambda \geq 0$, and LP strong duality gives $\max_{\lambda \geq 0} d_B(\lambda) = p^{\mathrm{LP}}(B)$ whenever $\mathcal{P}(B) \neq \emptyset$ (Proposition A.1). In the toy example $\min_{x \in [0,1]} x$ subject to $1/2 - x \leq 0$, setting $\lambda = 1$ gives $x + 1 \cdot (1/2 - x) = 1/2$ across the entire box, recovering the constrained optimum without branching (Section 3.3 and Appendix A.4). More generally, if a constraint row $i$ is strictly positive across the minimizing face $X_0 = \operatorname{argmin}_{x \in B} \underline{f}_0(x)$ ($v_i = \min_{x \in X_0}(a_i^\top x + c_i) > 0$), small steps along $e_i$ strictly raise $d_B$ at rate $v_i$ (**Theorem A.2**). Because the inner minimization is coordinate-separable, each evaluation of $d_B(\lambda)$ and its supergradient $Ax_\lambda + c$ at an inner minimizer $x_\lambda$ costs $O(mn)$ arithmetic per box without rerunning bound propagation. GEAR runs 20 projected Adam ascent steps and 500 batched GPU primal-dual hybrid gradient (PDHG) iterations with Ruiz equilibration (plus CPU LP solves on the 16 lowest-bound boxes when $mn \leq 2 \times 10^5$), evaluating $d_B(\lambda)$ with outward rounding every 50 PDHG steps and retaining the best certified bound ([Paper, Section 3.3 and Appendix B.5, Figure 12](https://arxiv.org/abs/2611.10487)).

2. **Incumbent-independent infeasibility certificates:** Whenever a nonnegative combination $\nu \geq 0$ satisfies

$$
\min_{x \in B} \nu^\top (Ax + c) > 0, \qquad \nu \geq 0 \implies \mathcal{F}(B) = \emptyset,
$$

the box $B$ cannot contain a feasible point and is discarded immediately ([Paper, Section 3.3, Equation (10)](https://arxiv.org/abs/2611.10487)). Unit vectors $\nu = e_i$ test each individual constraint in closed form; for the lowest-bound boxes whose constraints are only jointly inconsistent, $\nu$ is taken from the multipliers of a phase-one LP minimizing the maximum violation $\min\{t : Ax + c \leq t\mathbf{1},\; t \geq 0,\; x \in B\}$ on the CPU and verified by evaluating Equation (10) with outward rounding on the GPU ([Paper, Section 3.3 and Appendix B.5, Figure 12c](https://arxiv.org/abs/2611.10487)).

3. **Relaxed and complete domain clipping:** Adapting [Clip-and-Verify](/projects/neural-network-verification/) ([Zhou et al., 2025](https://neurips.cc/virtual/2025/poster/118862)) to nonlinear constraint rows, *relaxed clipping* uses $L_i = \min_{x \in B}(a_i^\top x + c_i)$ to tighten each coordinate interval in $O(mn)$ per sweep ([Paper, Section 3.3 and Appendix A.5, Equation (25)](https://arxiv.org/abs/2611.10487)):

$$
\overline{x}_j \leftarrow \min\!\left\{\overline{x}_j,\; \underline{x}_j - \frac{L_i}{a_{i,j}}\right\} \quad (a_{i,j} > 0),
\qquad
\underline{x}_j \leftarrow \max\!\left\{\underline{x}_j,\; \overline{x}_j - \frac{L_i}{a_{i,j}}\right\} \quad (a_{i,j} < 0).
$$

*Complete clipping* tightens intermediate graph intervals by maximizing $\min_{x \in B}[\underline{z}(x) + \beta^\top(Ax + c)]$ over $\beta \geq 0$—replacing coordinate descent with gradient descent when constraints are numerous, or sweeping up to 16 active rows through a fused double-precision piecewise-linear breakpoint kernel ([Paper, Section 3.3, Appendix A.5, and Appendix B.5](https://arxiv.org/abs/2611.10487)). Tighter intervals shrink the domains of downstream nonlinear operators on the next propagation pass, and any empty interval certifies $\mathcal{F}(B) = \emptyset$.

## Reuse geometry without growing the node state with depth

Because any child box $B_{\mathrm{child}} \subseteq B_{\mathrm{parent}}$ satisfies $\mathcal{F}(B_{\mathrm{child}}) \subseteq \mathcal{F}(B_{\mathrm{parent}})$, constraint linear bounds derived on a parent remain valid on its children ([Paper, Section 3.5 and Appendix A.3, Equation (18)](https://arxiv.org/abs/2611.10487)). GEAR forwards the parent's lower bound and constraint rows to both children so each child can be clipped immediately, before running its own bound propagation pass. Once the child computes fresh linear bounds, the new nonlinear rows replace the inherited ones while the original linear constraints stay shared. Consequently, **the active row set per box never grows with tree depth** ([Paper, Section 3.5 and Appendices A.3, A.11](https://arxiv.org/abs/2611.10487)).

4. **Multiplier-weighted smart branching ($\lambda$-SB):** Classical smart branching in neural verification splits each candidate coordinate and bounds both children, requiring $2n$ bound evaluations per box. In constrained NLP, a split can help either by raising the objective bound or by tightening a bottleneck constraint. GEAR scores every coordinate $j$ directly from the cached linear bounds and the best Lagrangian multipliers $\lambda_{\mathrm{best}}$ without bounding any child ([Paper, Section 3.5, Equation (12)](https://arxiv.org/abs/2611.10487)):

$$
S_j = w_j \left(\lvert a_{0,j}\rvert + \sum_{i=1}^m \lambda_i^{\mathrm{best}}\lvert a_{i,j}\rvert\right),
\qquad
w_j = \overline{x}_j - \underline{x}_j.
$$

Summing absolute coefficient magnitudes prevents opposing objective and constraint slopes from cancelling, and **Lemma A.5** shows that $S_j / 4$ upper-bounds the average one-step midpoint gain of the mixed row $a(\lambda)^\top x + c(\lambda)$ ([Paper, Appendix A.9, Equation (51)](https://arxiv.org/abs/2611.10487)). Maximization is restricted to coordinates with $w_j \geq \tau \max_k w_k$ ($\tau \in (0, 1]$) and paired with a count guard that forces a split along the relatively widest coordinate at least once every $n + 1$ splits, guaranteeing geometric box shrinkage $\max_j (w_j(B)/w_j^0) \leq 2^{-\lfloor k/(n(n+1))\rfloor}$ at depth $k$ and conditional finite $\varepsilon$-convergence (**Theorem A.8**, [Paper, Appendices A.9, A.13, and B.5, Equation (67)](https://arxiv.org/abs/2611.10487)).

## Keep primal recovery in the same batching regime

Strong lower bounds cannot prune by dominance ($\mathrm{LB}(B) \geq \mathrm{UB}$) without a good feasible incumbent. Calling a sequential KKT-factorization solver per box would bottleneck the GPU pipeline.

5. **Dual-guided projected ALM primal search:** GEAR searches for feasible incumbents across active boxes in parallel using the Hestenes–Powell–Rockafellar inequality-form augmented Lagrangian on the original nonlinear graph ([Paper, Section 3.4, Equation (11), and Appendix A.8, Algorithm 1](https://arxiv.org/abs/2611.10487)):

$$
\phi_\rho(x; \mu) = f_0(x) + \frac{\rho}{2}\sum_{i=1}^m \left[f_i(x) + \frac{\mu_i}{\rho}\right]_+^2 - \frac{\lVert\mu\rVert_2^2}{2\rho},
\qquad
[t]_+ = \max\{t, 0\}.
$$

Each outer iteration takes a fixed budget of projected Adam steps on $B$ using the exact gradient $\nabla_x \phi_\rho(x; \mu) = \nabla f_0(x) + \sum_{i=1}^m [\mu_i + \rho f_i(x)]_+ \nabla f_i(x)$ (Equation (43)), updates $\mu \leftarrow [\mu + \rho f(x)]_+$, and increases $\rho$ when the maximum constraint violation fails to shrink sufficiently. The dual evaluation warm-starts this primal search: GEAR initializes $x$ at $x_{\mathrm{best}} \in \operatorname{argmin}_{x \in B}[\underline{f}_0(x) + \lambda_{\mathrm{best}}^\top(Ax + c)]$—which already lies in the low-objective basin of the linear relaxation—and initializes $\mu^{(0)} = \kappa \lambda_{\mathrm{best}}$ so the first outer loop immediately weights the constraints that bottlenecked the dual bound ([Paper, Section 3.4 and Appendix A.8](https://arxiv.org/abs/2611.10487)). In the full solver (Appendix B.6, Figure 14), batched ALM-PGD (20,480 starts at the root; ~1,000 starts across a tree batch) is followed on the GPU by damped Gauss–Newton / Levenberg–Marquardt feasibility restoration (Equation (71)) and Mehrotra interior-point SQP polish (Equation (72)), alongside root CPU IPOPT and LP alternation. Every candidate updates the incumbent $P$ only after passing the unscaled feasibility gate $\max_i [f_i(\bar{x})]_+ \leq 10^{-6}$ on the original model (Equations (50) and (70)), keeping global soundness strictly decoupled from heuristic convergence ([Paper, Section 3.5, Table 4, and Appendix A.12](https://arxiv.org/abs/2611.10487)).

## Measure both coverage and the proof contribution of each component

The paper evaluates GEAR on **505 bounded continuous NLP instances**—**265 from `gams_world`** and **240 from `minlplib`** (`minlplib_linear`)—against six global solvers (`Gurobi`, `SCIP`, `BARON`, `Xpress`, `Couenne`, `LindoGlobal`) and six local solvers (`Knitro`, `IPOPT`, `CONOPT`, `LGO`, `MINOS`, `SNOPT`) through AMPL on one workstation (8-core 16-thread AMD Ryzen 9700X, 256 GB RAM, one NVIDIA RTX 5090) with a 180-second wall-clock limit, feasibility tolerance $10^{-6}$, and relative optimality gap $10^{-4}$ ([Paper, Section 4, Table 1, Figure 2, and Appendix C](https://arxiv.org/abs/2611.10487)):

| Solver | Type | `gams_world` Feas. / Solved (of 265) | `minlplib` Feas. / Solved (of 240) | Total Feas. / 505 | Total Solved / 505 | Solved (%) | Exclusive vs. GEAR (GEAR only / Baseline only) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GEAR** | **Global (GPU + CPU)** | **264 / 251** | **237** / 170 | **501** | **421** | **83.4%** | — |
| Gurobi | Global (commercial, 16T) | 254 / 227 | 218 / **174** | 472 | 401 | 79.4% | **38** / 18 |
| SCIP | Global (open-source, 16T) | 262 / 209 | 219 / 155 | 481 | 364 | 72.1% | **64** / 7 |
| BARON | Global (commercial) | 217 / 168 | 186 / 129 | 403 | 297 | 58.8% | **131** / 7 |
| Xpress | Global (commercial, 16T) | 218 / 176 | 173 / 120 | 391 | 296 | 58.6% | **138** / 13 |
| Couenne | Global (open-source) | 222 / 182 | 152 / 100 | 374 | 282 | 55.8% | **139** / 0 |
| LindoGlobal | Global (commercial) | 180 / 164 | 169 / 110 | 349 | 274 | 54.3% | **149** / 2 |
| *Global union (all 6)* | *6 global solvers combined* | — | — | — | *427* | *84.6%* | **16** / 22 |
| Knitro* | Local (matched to global D) | 263 / 231 | 225 / 170 | 488 | 401 | 79.4% | **37** / 17 |
| IPOPT* | Local (matched to global D) | 256 / 203 | 202 / 117 | 458 | 320 | 63.4% | **111** / 10 |
| CONOPT* | Local (matched to global D) | 259 / 205 | 205 / 91 | 464 | 296 | 58.6% | **133** / 8 |
| LGO* | Local (matched to global D) | 249 / 206 | 160 / 78 | 409 | 284 | 56.2% | **138** / 1 |
| MINOS* | Local (matched to global D) | 247 / 193 | 133 / 53 | 380 | 246 | 48.7% | **179** / 4 |
| SNOPT* | Local (matched to global D) | 235 / 182 | 133 / 51 | 368 | 233 | 46.1% | **191** / 3 |

GEAR certifies **421 of 505 instances (83.4%)** and returns a feasible point on **501**, exceeding Gurobi (**401**) by 20 instances and SCIP (**364**) by 57. Its lead is largest on `gams_world` (**251 / 265, 94.7%** vs. 227 for Gurobi and 209 for SCIP), while on `minlplib` GEAR (**170 / 240**) sits ahead of SCIP (**155**) and 4 instances behind Gurobi (**174**). Against the union of all six global solvers (**427** instances), GEAR certifies **16 instances that no global baseline solves** and misses **22**, bringing combined portfolio coverage to **443 / 505** ([Paper, Section 4.1, Table 1, and Figure 2](https://arxiv.org/abs/2611.10487)).

The cumulative ablation in **Table 2** ([Paper, Section 4.2, page 9](https://arxiv.org/abs/2611.10487)) shows how each component contributes across all 505 instances under the 180-second limit:

| Cumulative configuration (505 instances, 180 s) | Feasible | Optimal* | Solved | Step gain in Solved |
| --- | ---: | ---: | ---: | ---: |
| Bound propagation + PGD (verification baseline) | 95 | 56 | 0 | — |
| + presolve and CPU solving components | 480 | 396 | 219 | +219 |
| + projected ALM primal search (GPU) | 496 | 471 | 269 | +50 |
| + constraint-aware dual bounds and clipping (GPU) | 498 | 472 | 366 | +97 |
| + lifted and refined operator relaxation (GPU) | 501 | 478 | 410 | +44 |
| + $\lambda$-SB branching (full GEAR) | 501 | 478 | 421 | +11 |

Here `Optimal*` counts runs whose returned feasible point is already within the $10^{-4}$ relative gap of the best known optimum verified by any global solver, whereas `Solved` requires GEAR's own certified lower bound $D$ to close Equation (13). Two structural facts stand out ([Paper, Section 4.2](https://arxiv.org/abs/2611.10487)):

1. **Why the verification pipeline alone solves 0 constrained NLPs:** Standard input branch-and-bound with a box PGD attack (`Bound propagation + PGD`) solves **0** instances. PGD searches only for a counterexample on the box without respecting $f_i(x) \leq 0$; on the 95 unconstrained (box-only) instances where every box point is feasible, PGD reaches the optimal value on 56, but the verification pipeline keeps no verified feasible incumbent to serve as a dominance-pruning threshold.
2. **Separating primal discovery from proof completion:** Once presolve (`+219`, closing 72 at the root and 147 in the tree) and GPU projected ALM (`+50`) are active, GEAR already holds an optimal feasible point on **471 of 505** instances, yet only **269** are certified. The next three rows—constraint-aware dual bounds and clipping (**+97**, the single largest GPU gain), lifted and refined operator relaxations (**+44**), and $\lambda$-SB branching (**+11**)—raise `Solved` from **269 to 421 (+152)** while adding only **7** optimal points ($471 \to 478$). Their primary role is **closing the dual proof**, turning already-found optimal incumbents into certified global optima.

## Choose a relaxation that can sustain the workload

Propagating a single affine bound per operator output keeps backward passes regular and tensorized, at the price of dropping simultaneous multi-facet closure on a single box. GEAR narrows that local relaxation gap in three ways before relying on spatial refinement ([Paper, Section 3.3, Appendix A.7, Table 6, and Appendices B.1–B.4](https://arxiv.org/abs/2611.10487)):

- **Coefficient-aware McCormick selection for bilinear products ($xy$):** A lifted LP keeps all four McCormick facets (Equations (38)–(39)), whereas backward LBP keeps one plane per output. Instead of fixing one McCormick vertex, GEAR evaluates both candidate lower planes (for backward weight $\gamma \geq 0$) or both upper planes (for $\gamma < 0$) by closed-form box concretization (Equation (37)) and selects the tighter one per element. On $[-5, -1] \times [-5, -2]$, this raises the product lower bound from $-10$ to $+2$ (Figure 4a,b).
- **Guarded 8-plane fused division ($x/y$):** Rather than decomposing $x/y = x \cdot (1/y)$ through an intermediate variable, GEAR scores four corner tangents (Equation (40)) and four three-corner secant planes directly on sign-fixed guarded denominator intervals ($y_L \geq \delta_{\mathrm{div}} > 0$ or $y_U \leq -\delta_{\mathrm{div}} < 0$), accepting a candidate plane $p(x,y)$ only after a quadratic-bilinear residual check certifies $R_p(x,y) = y p(x,y) - x \leq 0$ across the rectangle (Equation (41)). On $[1, 3] \times [0.5, 2]$, fused division lifts the lower bound from $0.32$ to $0.50$ (Figure 4c,d).
- **Parallel secant-tangent univariate bounds and exact presolve rewrites:** For convex/concave univariate operators ($\log$, $\sqrt{\cdot}$, $1/x$, $\exp$, $t^p$), choosing the tangent point $t_0$ whose slope matches the secant slope, $h'(t_0) = (h(\beta) - h(\alpha))/(\beta - \alpha)$ (Equations (32)–(36)), cuts the tangent-side gap by **26% for $\log$**, **27% for $\sqrt{\cdot}$**, **31% for $1/x$**, and **23% for $\exp$** relative to midpoint tangents ([Paper, Appendix A.7, Figure 3](https://arxiv.org/abs/2611.10487)), while trigonometric intervals are localized at inflection points. At presolve and the root, exact graph rewrites ($u \cdot u \to \mathrm{Sqr}(u)$, completing squares, quadratic eigenbasis and log-coordinate posynomial rewrites; Figures 6–8), root certificates (Appendix B.3), and a lifted linear relaxation with McCormick, reduced RLT, OBBT, and leaf certificate transfer (Appendix B.4, Figures 9–11) close **293 of the 421** solved instances at the root, leaving **128** to close in the GPU branch-and-bound tree (with a median of **318** boxes and at least $10^5$ boxes on five instances; [Paper, Section 4.1 and Appendix B.2](https://arxiv.org/abs/2611.10487)).

The breakdown by problem class in **Table 10** and the 40-instance disagreement audit against `Gurobi` and `SCIP` in **Table 11** ([Paper, Section 4.1 and Appendices D.1–D.2, pages 50–51](https://arxiv.org/abs/2611.10487)) locate the exact boundary between GPU-batched bound propagation and CPU LP branch-and-bound:

| Problem class (Table 10) | Instances | GEAR | SCIP | Gurobi | 6-Solver Global Union | GEAR-only vs. Union | Union-only vs. GEAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\leq 5$ variables | 183 | **182** | 171 | 177 | 182 | 0 | 0 |
| $6\text{–}20$ variables | 151 | **145** | 118 | 127 | 140 | **8** | 3 |
| $21\text{–}100$ variables | 80 | **66** | 58 | **66** | 71 | 3 | 8 |
| $101\text{–}1000$ variables | 65 | 28 | 17 | **29** | 32 | 5 | 9 |
| $> 1000$ variables | 26 | 0 | 0 | **2** | 2 | 0 | 2 |
| Box constraints only ($m = 0$) | 95 | **92** | 81 | 81 | 89 | 3 | 0 |
| $1\text{–}20$ constraints | 235 | **230** | 197 | 217 | 226 | 6 | 2 |
| $> 20$ constraints | 175 | 99 | 86 | **103** | 112 | 7 | **20** |
| **Transcendental terms** (`exp`, `log`, `sqrt`, trig) | 147 | **131** | 100 | 110 | 118 | **16** | 3 |
| Polynomial / rational only | 358 | 290 | 264 | **291** | 309 | 0 | 19 |

Three operational lessons emerge from Tables 10 and 11:

1. **The benefit is hard-tail coverage, not easy-instance latency:** On the 405 instances certified by both GEAR and at least one global baseline, the fastest CPU solver needs a median of **0.03–0.06 s** across size classes, whereas GEAR pays a few seconds of graph-compilation and GPU-initialization overhead (**4–29 s** median). GEAR overtakes `LindoGlobal`, `Couenne`, `BARON`, and `Xpress` at **6–9 s**, `SCIP` at **15 s**, and `Gurobi` at **35 s** ([Paper, Section 4.1 and Appendix D.1](https://arxiv.org/abs/2611.10487)).
2. **GEAR dominates on low- to medium-dimensional problems with composite transcendental nonlinearities:** On instances with at most 20 constraints, GEAR certifies **322 / 330 (97.6%)** versus 298 for Gurobi and 278 for SCIP. On instances with transcendental terms, GEAR certifies **131 / 147 (89.1%)** versus 110 for Gurobi and 100 for SCIP—including **all 16 instances certified by none of the six global solvers**. In the 22-instance win audit against Gurobi and SCIP (Table 11), 16 wins come from stronger bounds on log-coordinate posynomials (`hs101`, `hs102`, `schitt_s380`), polar polygon diameter bounds (`polygon25`–`100`), symmetric log-barrier collapsing (`Paviani`, `hs110`), and loose baseline factorable relaxations (`pricing050`, `Shekelfox10`), while 6 come from baseline solutions that violate original constraints by $1.4 \times 10^{-6}$ to $0.16$ upon AMPL re-evaluation (GEAR's largest violation is $9.9999994 \times 10^{-7}$; Appendix B.7 and Table 11).
3. **GPU batching does not replace RLT and warm-started simplex on large bilinear networks:** On problems with $> 20$ constraints (pooling, wastewater, water networks, and `kall` circle packing), 20 of the 22 Global-union-only wins occur, and GEAR solves no instance with $> 1{,}000$ variables. On 16 of the 18 instances where Gurobi or SCIP beats GEAR (Table 11), GEAR's lifted McCormick LP matches Gurobi's root bound, but Gurobi re-solves warm-started node LPs with RLT and cuts at **$2.9 \times 10^3$ to $4.0 \times 10^4$ nodes/s** versus **1.1–207 nodes/s** (median 27) in GEAR's CPU lifted-LP tree, while the batched GPU bounds close a median of only **0.74%** of the remaining gap on those bilinear-heavy leaves ([Paper, Appendix D.2, Table 11](https://arxiv.org/abs/2611.10487)).

Memory shapes this boundary as well: storing the dense coefficient tensor $A \in \mathbb{R}^{K \times (m+1) \times n}$ together with intermediate graph intervals scales with $K(m+1)n$, so the dynamic batch controller doubles $K$ from $k_0 = 2$ only while $< 45\%$ of GPU memory is reserved and halves $K$ upon out-of-memory ([Paper, Appendix B.5](https://arxiv.org/abs/2611.10487)). Within its scope—bounded continuous NLPs over supported factorable operators—GEAR establishes a complementary GPU regime for global optimization, extending [linear bound propagation and domain clipping from neural network verification](/projects/neural-network-verification/) into a complete primal-dual global solver.
