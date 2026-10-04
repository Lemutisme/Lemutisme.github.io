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

A global optimizer may need to examine millions of subdomains before it can certify a solution. Its runtime depends on both the cost of processing each domain and the number of domains that survive its bounds. GEAR improves both: it processes domains in GPU batches, then uses the resulting linear coefficients to make each batch remove more of the search space.

This requires the solver's operations to share a representation. For every subdomain, a bound on the objective must interact with nonlinear constraints, feasible solution search, and branching. GEAR makes affine lower bounds the common interface between these operations.

## Turn the search frontier into a tensor workload

GEAR compiles the objective and all constraints into one multi-output computation graph. A backward propagation pass is vectorized over both outputs and branch-and-bound nodes. Each node retains its own bounds while sharing the graph's computational structure.

The resulting coefficient tensors support several operations without constructing a separate LP solver for each node:

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

The paper's cumulative ablation makes the effect on search visible:

| Configuration, all at an 80-second cap | Solved / 505 | Average visited domains |
| --- | ---: | ---: |
| Basic linear bound propagation + ALM-PGD | 248 | 7.94 million |
| Through complete clipping | 361 | 1.58 million |
| Full GEAR | 383 | 1.30 million |

The full method solves more instances while reducing the average visited-domain count by about 84%. These measurements connect scalability to the quality of search, alongside the ability to process nodes in parallel. [Paper, Table 2](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

## Extend the set of problems that can be certified

At a 180-second cap, GEAR solves **399 of 505** bounded continuous instances from GAMS and MINLPLib, compared with **334** for the paired SCIP-Nonlinear baseline. GEAR adds **104** instances on which SCIP reaches the time limit. SCIP adds 39 instances, and their combined coverage is 438. GEAR uses the CPU and GPU; SCIP uses the CPU. [Paper, Section 4 and Appendix C](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

These exclusive solves show where the architecture is useful: difficult search can benefit from processing many graph-based relaxations and reusing their geometry throughout the solver.

The single affine bound propagated through each nonlinear operation keeps this computation regular, with a tradeoff in relaxation strength. A full LP relaxation can retain several useful facets simultaneously, which helps SCIP close some problems at the root. GEAR's advantage depends on how well bound quality and GPU throughput work together on the problem's structure.

GEAR targets bounded continuous programs over supported factorable operators. Pruning uses valid lower bounds, and primal candidates are checked against the original constraints before they update the incumbent. The [research note](/blog/gear/) develops the scaling argument, the coefficient reuse, and the memory tradeoffs in more detail; the [verification project](/projects/neural-network-verification/) traces the bound propagation and clipping machinery this solver builds on.
