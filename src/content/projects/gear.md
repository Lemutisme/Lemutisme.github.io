---
title: "GEAR"
tagline: "Scale global optimization by batching proof work and reusing linear bounds throughout search."
description: "GEAR combines GPU throughput with stronger pruning: one batched propagation pass supplies constraint relaxations, Lagrangian dual bounds, infeasibility certificates, primal seeds, and branching scores. It certifies 421 of 505 bounded continuous NLP instances within 180 seconds, more than each of six global solvers including Gurobi, BARON, and SCIP."
status: "NeurIPS 2026"
period: "2026"
citationYear: 2026
order: 0
tags:
  - Global Optimization
  - Nonlinear Programming
  - Linear Bound Propagation
  - GPU Computing
links:
  - label: "arXiv"
    url: "https://arxiv.org/abs/2611.10487"
  - label: "PDF"
    url: "https://arxiv.org/pdf/2611.10487"
  - label: "OpenReview"
    url: "https://openreview.net/forum?id=3WdBmyLLLC"
  - label: "Research note"
    url: "https://www.duo-zhou.com/blog/gear/"
bibtex: |
  @article{zhou2026gear,
    author        = {Zhou, Duo and Chen, Hesun and Zhong, Xiangru and Hanasusanto, Grani A. and Zhang, Huan},
    title         = {{GEAR}: A {GPU}-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation},
    journal       = {arXiv preprint arXiv:2611.10487},
    year          = {2026},
    eprint        = {2611.10487},
    archivePrefix = {arXiv},
    primaryClass  = {math.OC},
    url           = {https://arxiv.org/abs/2611.10487}
  }
---

**GEAR: A GPU-Accelerated Global Solver for Bounded Continuous Nonlinear Programs via Linear Bound Propagation**

**Duo Zhou**, Hesun Chen, Xiangru Zhong, Grani A. Hanasusanto, Huan Zhang

Accepted to **NeurIPS 2026**. [arXiv](https://arxiv.org/abs/2611.10487) · [PDF](https://arxiv.org/pdf/2611.10487) · [OpenReview](https://openreview.net/forum?id=3WdBmyLLLC) · [Read the research note](/blog/gear/)

## Two ways to make global search scale

A global optimizer may need to examine thousands or millions of subdomains before it can certify a solution. Its runtime depends on both the cost of processing each domain and the number of domains that survive its bounds:

$$
T_{\mathrm{search}} \approx N_{\mathrm{visited}} \frac{T_{\mathrm{batch}}}{K} + T_{\mathrm{queue}}.
$$

Classical global NLP solvers reduce $N_{\mathrm{visited}}$ by solving polyhedral LP relaxations at each node, strengthened with McCormick envelopes, optimization-based bound tightening (OBBT), and Reformulation-Linearization Technique (RLT) cuts. Variable cut counts and irregular active sets keep those per-node subproblems on the CPU. GEAR makes the opposite architectural trade: it propagates a single affine lower bound $\underline{f}_i(x) = a_i^\top x + c_i \le f_i(x)$ per output so that a frontier of $K$ heterogeneous boxes shares fixed tensor shapes $K \times (m+1) \times n$ and $K \times (m+1)$. It then recovers pruning and primal-search strength by making every stage of node processing reuse those same linear coefficients ([Paper, Sections 1 and 3](https://arxiv.org/abs/2611.10487)).

## Turn the search frontier into a tensor workload

GEAR compiles the objective $f_0$ and all $m$ shifted inequality constraints $f_1, \ldots, f_m \le 0$ into one multi-output computation graph $F(x) = (f_0(x), f_1(x), \ldots, f_m(x))$. Linear constraints $a_i^\top x + c_i \le 0$ need no relaxation and are substituted exactly by linear bound propagation (LBP). A backward propagation pass is vectorized over both the $m+1$ outputs and the $K$ branch-and-bound boxes: each box retains its own interval bounds and linear coefficients while sharing the graph's computational structure.

One set of propagated linear bounds serves five roles across the solver without constructing a separate LP model for each operation ([Paper, Figure 1 and Sections 3.2–3.5](https://arxiv.org/abs/2611.10487)):

| Reuse of the same linear bounds | Effect on global search |
| --- | --- |
| Constraint relaxations and domain clipping | Relaxed clipping shrinks input box coordinates in $O(mn)$ per box; complete clipping tightens intermediate intervals over the linearized constraint polyhedron $\mathcal{P}(B) = \{x \in B : Ax + c \le 0\}$. |
| Lagrangian lower bound $d_B(\lambda)$ | Dualizes $Ax + c \le 0$ over the box $B = [\underline{x}, \overline{x}]$ via $d_B(\lambda) = \min_{x \in B}[\underline{f}_0(x) + \lambda^\top (Ax + c)]$ for $\lambda \ge 0$, evaluated in closed form in $O(mn)$ per box and tightened by projected Adam and batched GPU PDHG. |
| Incumbent-independent infeasibility certificates | Certifies $\mathcal{F}(B) = \emptyset$ whenever $\min_{x \in B} \nu^\top (Ax + c) > 0$ for $\nu \ge 0$, using unit vectors $\nu = e_i$ or Farkas multipliers from a phase-one violation LP. |
| Dual-guided projected ALM primal search | Starts batched first-order augmented Lagrangian updates from the Lagrangian inner minimizer $x_{\mathrm{best}}$ and initializes ALM multipliers $\mu$ by rescaling $\lambda_{\mathrm{best}}$. |
| Multiplier-weighted smart branching ($\lambda$-SB) | Scores coordinate $j$ by $S_j = (\overline{x}_j - \underline{x}_j)(\lvert a_{0,j}\rvert + \sum_{i=1}^m \lambda_i^{\mathrm{best}}\lvert a_{i,j}\rvert)$ to pick midpoint splits without bounding candidate children. |

Projected augmented Lagrangian updates, Gauss–Newton feasibility restoration, and Mehrotra SQP polish also execute as batched GPU computations ([Paper, Section 3.4 and Appendix B.6](https://arxiv.org/abs/2611.10487)), keeping primal recovery in the same throughput regime as dual bounding.

## Stronger bounds make throughput useful

An inexpensive bound becomes useful only when it can retire a domain. Inherited constraint rows remain valid on child boxes, allowing each child to be clipped immediately before running its own propagation pass; fresh nonlinear rows then replace the inherited ones so per-box memory does not grow with tree depth ([Paper, Section 3.5 and Appendix A.3](https://arxiv.org/abs/2611.10487)).

Because LBP keeps one affine plane per operator output, local envelope quality directly determines downstream tightness ([Paper, Section 3.3 and Appendix A.7](https://arxiv.org/abs/2611.10487)):

- **Bilinear products ($xy$):** Instead of fixing one McCormick vertex, GEAR compares the two lower (or upper) McCormick facets by closed-form box concretization according to the backward coefficient sign. On $[-5, -1] \times [-5, -2]$, coefficient-aware selection raises the lower bound from $-10$ to $+2$ (Figure 4a,b).
- **Quotients ($x/y$):** On sign-fixed guarded denominator intervals, GEAR scores an 8-plane candidate pool (four corner tangents and four three-corner secant planes) directly on $x/y$ and accepts a plane only after a quadratic-bilinear residual check $y p(x,y) - x \le 0$. On $[1, 3] \times [0.5, 2]$, fusing division lifts the lower bound from $0.32$ to $0.50$ (Figure 4c,d).
- **Univariate and trigonometric primitives:** Parallel secant-tangent relaxations match the tangent slope to the secant slope, cutting the tangent-side bound gap by 26% for $\log$, 27% for $\sqrt{\cdot}$, 31% for $1/x$, and 23% for $\exp$ over midpoint tangents ([Paper, Figure 3](https://arxiv.org/abs/2611.10487)), while exact graph rewrites (completing squares, $u \cdot u \to \mathrm{Sqr}(u)$, quadratic eigenbasis and log-coordinate posynomial rewrites) and a lifted linear relaxation strengthen root and tree certificates ([Paper, Appendices B.1–B.4](https://arxiv.org/abs/2611.10487)).

<figure>
  <img src="/figures/gear/envelopes.png" alt="Four surface plots from Figure 4 of the paper. A fixed-vertex product envelope has lower bound -10 on [-5, -1] x [-5, -2]. A coefficient-aware envelope on the same box has lower bound 2. A decomposed quotient envelope has lower bound 0.32 on [1, 3] x [0.5, 2]. A fused quotient envelope has lower bound 0.50.">
  <figcaption>Linear lower bounds for two bivariate operators (Paper, Figure 4). On [-5, -1] x [-5, -2], selecting the McCormick plane with the larger box minimum raises the product bound from -10 to 2. On [1, 3] x [0.5, 2], a guarded fused quotient plane raises the bound from 0.32 (reciprocal-product decomposition) to 0.50.</figcaption>
</figure>

The paper's cumulative ablation across all 505 instances under the 180-second limit separates primal discovery from dual certification ([Paper, Section 4.2, Table 2](https://arxiv.org/abs/2611.10487)):

| Configuration (505 instances, 180 s limit) | Feasible | Optimal* | Solved | Change in Solved |
| --- | ---: | ---: | ---: | ---: |
| Bound propagation + PGD (verification baseline) | 95 | 56 | 0 | — |
| + presolve and CPU solving components | 480 | 396 | 219 | +219 |
| + projected ALM primal search (GPU) | 496 | 471 | 269 | +50 |
| + constraint-aware dual bounds and clipping (GPU) | 498 | 472 | 366 | +97 |
| + lifted and refined operator relaxation (GPU) | 501 | 478 | 410 | +44 |
| + $\lambda$-SB branching (full GEAR) | 501 | 478 | 421 | +11 |

The standard neural network verification pipeline (`Bound propagation + PGD`) solves **0** instances: a box PGD attack looks for a counterexample without enforcing nonlinear constraints, so even though it hits the optimal value on 56 of the 95 box-constrained problems, it keeps no verified feasible incumbent as a pruning threshold. Once presolve and GPU projected ALM are added, GEAR already holds a feasible point within $10^{-4}$ of the best known optimum (`Optimal*`) on **471 of 505** instances, with 269 certified. The remaining components—constraint-aware dual bounds and clipping (**+97**, the single largest GPU step), lifted and refined operator relaxations (**+44**), and $\lambda$-SB branching (**+11**)—raise the certified solved count from **269 to 421 (+152)** while adding only **7** optimal points ($471 \to 478$). They finish the optimality proof rather than merely finding better points.

## Where GPU-batched bounds beat per-node LP strength, and where they do not

Across 505 bounded continuous NLP instances (**265 from `gams_world`** and **240 from `minlplib`**) on a single workstation (AMD Ryzen 9700X, 256 GB RAM, NVIDIA RTX 5090) under a 180-second limit, absolute feasibility tolerance $10^{-6}$, and relative optimality gap $(P - D)/\max(1, \lvert P\rvert, \lvert D\rvert) \le 10^{-4}$, GEAR certifies **421 of 505 instances (83.4%)** and returns a feasible point on **501**—more than each of the six global solvers and six local solvers compared ([Paper, Section 4.1, Table 1](https://arxiv.org/abs/2611.10487)):

| Solver | Category | `gams_world` Solved / 265 | `minlplib` Solved / 240 | Total Feasible / 505 | Total Solved / 505 | Solved (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **GEAR** | **Global (GPU + CPU)** | **251** | 170 | **501** | **421** | **83.4%** |
| Gurobi | Global (commercial) | 227 | **174** | 472 | 401 | 79.4% |
| SCIP | Global (open-source) | 209 | 155 | 481 | 364 | 72.1% |
| BARON | Global (commercial) | 168 | 129 | 403 | 297 | 58.8% |
| Xpress | Global (commercial) | 176 | 120 | 391 | 296 | 58.6% |
| Couenne | Global (open-source) | 182 | 100 | 374 | 282 | 55.8% |
| LindoGlobal | Global (commercial) | 164 | 110 | 349 | 274 | 54.3% |
| Knitro* | Local (matched to global bound) | 231 | 170 | 488 | 401 | 79.4% |
| IPOPT* | Local (matched to global bound) | 203 | 117 | 458 | 320 | 63.4% |

Against the strongest global baseline, **Gurobi (401)**, GEAR certifies **38** instances that Gurobi does not while Gurobi certifies **18** that GEAR does not; against **SCIP (364)**, the exclusive split is **64 to 7**. Across all six global solvers individually, GEAR certifies **38–149** instances that each baseline misses against **0–18** in the reverse direction. Against the **union of all six global solvers** (**427** instances), GEAR certifies **16** instances solved by none of the six global solvers and misses **22**, bringing combined portfolio coverage to **443 of 505** ([Paper, Section 4.1, Figure 2](https://arxiv.org/abs/2611.10487)).

<figure>
  <img src="/figures/gear/cactus.png" alt="Three-panel plot from Figure 2 of the paper. Panel (a) shows cactus solve-time curves for GEAR (421) and six global solvers: Gurobi (401), SCIP (364), BARON (297), Xpress (296), Couenne (282), and LindoGlobal (274). Panel (b) shows cactus curves for GEAR (421) and six local solvers: Knitro* (401), IPOPT* (320), CONOPT* (296), LGO* (284), MINOS* (246), and SNOPT* (233). Panel (c) shows pairwise exclusive solves between GEAR and each baseline, including 38 vs 18 against Gurobi, 64 vs 7 against SCIP, and 16 vs 22 against the union of all six global solvers.">
  <figcaption>Solve times and exclusive solves at 180 seconds (Paper, Figure 2). Panels (a, b) compare GEAR against six global solvers and six local solvers (credited only when their feasible objective matches a globally certified bound). Panel (c) shows instances solved only by GEAR (right, teal) versus only by the baseline (left, brown); against the union of all six global solvers, GEAR adds 16 exclusive certifications and misses 22.</figcaption>
</figure>

The problem-class breakdown ([Paper, Section 4.1 and Appendix D, Tables 10–11](https://arxiv.org/abs/2611.10487)) shows where the GPU-batched regime wins and where classical LP branch-and-bound still dominates:

- **Low- to medium-dimensional problems with composite transcendental nonlinearities:** GEAR pays a startup cost of a few seconds for graph compilation and GPU initialization, overtaking LindoGlobal, Couenne, BARON, and Xpress at **6–9 s**, SCIP at **15 s**, and Gurobi at **35 s**. Its lead comes from the hard tail: GEAR certifies **322 of 330 instances (97.6%)** with at most 20 constraints (vs. 298 for Gurobi and 278 for SCIP) and **131 of 147 instances (89.1%)** containing transcendental terms (vs. 110 for Gurobi and 100 for SCIP). **All 16 instances certified by GEAR and by none of the six global solvers contain exponential, logarithmic, or trigonometric terms.** GEAR closes **293 of its 421** solved instances at the root (including 12 of those 16) and **128** in the GPU branch-and-bound tree (median 318 boxes, with at least $10^5$ boxes on five instances).
- **Large structured bilinear models:** On instances with more than 20 constraints (pooling, wastewater, water networks, and `kall` circle packing), Gurobi certifies 103 of 175 against 99 for GEAR, and 20 of the 22 Global-union-only instances fall into this class. On the 16 bilinear QCQPs where Gurobi or SCIP wins and GEAR does not, Gurobi re-solves warm-started LPs with RLT and cuts at $2.9 \times 10^3$ to $4.0 \times 10^4$ nodes per second, whereas GEAR solves no instance with more than 1,000 variables.

GEAR targets bounded continuous programs over supported factorable operators; mixed-integer and unbounded programs are outside its scope. Pruning uses conservatively evaluated lower bounds and Farkas certificates, and primal candidates update the incumbent only after direct verification on the original constraints at $10^{-6}$. The [research note](/blog/gear/) develops the dual-primal coupling, the operator envelopes, the ablation analysis, and the structural regime boundary in detail; the [verification project](/projects/neural-network-verification/) traces the linear bound propagation and domain clipping machinery this solver builds on.
