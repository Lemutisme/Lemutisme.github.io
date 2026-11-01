---
title: "GEAR"
tagline: "Scale global optimization by batching proof work and reusing linear bounds throughout search."
description: "GEAR combines GPU throughput with stronger pruning: one batched propagation pass supplies constraint geometry, dual bounds, primal seeds, and branching scores. It solves 399 of 505 bounded continuous NLP instances within 180 seconds."
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
  - label: "Download PDF"
    url: "https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing"
  - label: "OpenReview"
    url: "https://openreview.net/forum?id=3WdBmyLLLC"
  - label: "Research note"
    url: "https://www.duo-zhou.com/blog/gear/"
bibtex: |
  @inproceedings{zhou2026gear,
    author    = {Zhou, Duo and Chen, Hesun and Zhong, Xiangru and Hanasusanto, Grani A. and Zhang, Huan},
    title     = {{GEAR}: A {GPU}-Accelerated Global Solver for Nonlinear Programs via Linear Bound Propagation},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2026},
    url       = {https://openreview.net/forum?id=3WdBmyLLLC}
  }
---

**GEAR: A GPU-Accelerated Global Solver for Nonlinear Programs via Linear Bound Propagation**

**Duo Zhou**, Hesun Chen, Xiangru Zhong, Grani A. Hanasusanto, Huan Zhang

Accepted to **NeurIPS 2026**. [Download the PDF](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing) · [OpenReview](https://openreview.net/forum?id=3WdBmyLLLC) · [Read the research note](/blog/gear/)

## Two ways to make global search scale

A global optimizer may need to examine millions of subdomains before it can certify a solution. Its runtime depends on both the cost of processing each domain and the number of domains that survive its bounds:

$$T_{\mathrm{search}} \approx N_{\mathrm{visited}} \frac{T_{\mathrm{batch}}}{K} + T_{\mathrm{queue}}.$$

Classical solvers reduce $N_{\mathrm{visited}}$ by solving polyhedral LP relaxations at each node, but variable cut counts and irregular active sets keep those subproblems on the CPU. GEAR makes the opposite architectural trade: it propagates a single affine lower bound per nonlinear operator so that a frontier of $K$ heterogeneous boxes shares a fixed tensor shape $K \times (m+1) \times n$. It then recovers relaxation strength by making every stage of node processing reuse those same linear coefficients.

## Turn the search frontier into a tensor workload

GEAR compiles the objective and all $m$ constraints into one multi-output computation graph. A backward propagation pass is vectorized over both outputs and branch-and-bound nodes. Each node retains its own interval bounds and linear coefficients while sharing the graph's computational structure.

The resulting coefficient tensors support the rest of the solver without constructing or solving a separate LP at each node:

| Reuse of the same coefficients | Effect on scalability |
| --- | --- |
| Necessary linear inequalities for nonlinear constraints | Clip boxes and eliminate infeasible regions before further search. |
| Nonnegative mixtures of objective and constraint bounds | Tighten the objective lower bound through tensor updates on cached rows. |
| A relaxation minimizer for primal initialization | Start batched feasible-point search from a point suggested by the current bound. |
| Constraint-weighted branching scores | Use the dual's information to choose splits without solving a separate relaxation for every candidate split. |

Projected augmented Lagrangian updates also run as first-order GPU computations. Keeping primal recovery in the same batching regime lets the bound computation's throughput carry through to the rest of the node processing.

## Stronger bounds make throughput useful

An inexpensive bound becomes valuable when it can retire a domain. GEAR uses nonlinear constraint rows to strengthen the objective bound and guide clipping. Tighter envelopes for products, quotients, and other NLP operators improve the coefficients consumed by all downstream operations.

Inherited constraint rows remain valid on child boxes. After a child computes its own bounds, locally derived rows replace the inherited rows. This carries useful geometry into the next batch while keeping the active row set bounded with respect to search depth.

The same propagation is also where the bound gets its strength. A fixed-vertex envelope for a product can certify a lower bound of −10 on a box where a coefficient-aware envelope certifies +2. For a quotient, fusing the reciprocal with the numerator lifts the lower bound from 0.32 to 0.50. Those are local gaps. They matter because every later clip, dual mix, and branch score consumes the same coefficients.

<figure>
  <img src="/figures/gear/envelopes.png" alt="Four surface plots. A fixed-vertex product envelope has lower bound -10. A coefficient-aware envelope on the same box has lower bound 2. A decomposed quotient envelope has lower bound 0.32. A fused quotient envelope has lower bound 0.50.">
  <figcaption>Lower envelopes for a product and a quotient. The coefficient-aware and fused forms are the ones GEAR propagates. A tighter envelope is a higher floor under the same box, and that floor is reused everywhere downstream.</figcaption>
</figure>

The paper's cumulative ablation makes the effect on search visible:

| Configuration, all at an 80-second cap | Solved / 505 | Average visited domains |
| --- | ---: | ---: |
| Basic linear bound propagation + ALM-PGD | 248 | 7.94 million |
| + Constraint-aware objective bounds and clipping | 361 | 1.58 million |
| + Refined operator envelopes and branching | 381 | 1.04 million |
| Full GEAR (+ linear-constraint ALM) | 383 | 1.30 million |

The full method solves 135 more instances while cutting the average visited-domain count by about 84%. Constraint-aware dual mixing and domain clipping account for the fivefold drop in search volume; refined operator envelopes and constraint-weighted branching push the domain count down further while solving 20 more problems. [Paper, Table 2](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

## Where throughput beats root strength, and where it does not

At a 180-second cap, GEAR solves **399 of 505** bounded continuous instances from GAMS and MINLPLib, compared with **334** for the paired SCIP-Nonlinear baseline. In that matched comparison, 295 are solved by both, 104 by GEAR only, and 39 by SCIP only. Their union is 438. GEAR uses the CPU and GPU; SCIP uses the CPU.

<figure>
  <img src="/figures/gear/cactus.png" alt="A cactus plot of runtime against solved instances for GEAR and several global solvers, and a bar chart of pairwise exclusive solves. GEAR's curve reaches 399 instances. The bars show instances solved by GEAR only against each other solver.">
  <figcaption>Coverage at 180 seconds. The cactus sorts instances by the time each solver needed. The bars are pairwise exclusive solves: orange is solved by GEAR and not by that solver, and the other color is the reverse. The controlled SCIP pair in the text, 104 against 39, is the matched comparison. The bars place that kind of non-overlap against the rest of the portfolio.</figcaption>
</figure>

The split between the 104 GEAR-only solves and the 39 SCIP-only solves reflects the core relaxation trade-off. A polyhedral LP relaxation can retain several facets of a nonlinear constraint at once; 23 of SCIP's 39 exclusive solves close at the root node, where parallel tree search has little work to exploit. GEAR propagates a single affine bound per operator to keep backward passes regular and tensorized, and wins on the 104 instances where certification requires exploring and pruning a large search tree.

GEAR targets bounded continuous programs over supported factorable operators. Pruning uses valid lower bounds, and primal candidates are checked against the original constraints before they update the incumbent. The [research note](/blog/gear/) develops the scaling argument, the coefficient reuse, and the memory tradeoffs in more detail; the [verification project](/projects/neural-network-verification/) traces the bound propagation and clipping machinery this solver builds on.
