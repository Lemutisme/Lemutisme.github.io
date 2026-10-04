---
title: "GEAR: Scaling Global Optimization with Reusable Linear Bounds"
description: "GEAR makes global search scale in two ways: GPU batches reduce the effective cost of processing a domain, while reusable constraint-aware bounds reduce how many domains the solver needs to visit."
publishedAt: 2026-10-03
tags:
  - Global Optimization
  - Nonlinear Programming
  - Linear Bound Propagation
  - GPU Computing
draft: false
---

**GEAR: A GPU-Accelerated Global Solver for Nonlinear Programs via Linear Bound Propagation** · **NeurIPS 2026**

**Duo Zhou**, Hesun Chen, Xiangru Zhong, Grani A. Hanasusanto, Huan Zhang

[Download the PDF](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing) · [OpenReview](https://openreview.net/forum?id=3WdBmyLLLC) · [Project page](/projects/gear/)

Global optimization becomes expensive when a solver must inspect millions of regions before it can certify a solution. Each region needs an objective lower bound, a feasibility check, a search for a feasible point, and a decision about the next split. GEAR makes these operations consume common affine coefficients, allowing many regions to be processed as one GPU workload and making the information from each propagation pass useful throughout the search.

There are two sources of scalability here: **lower the effective cost of processing a domain, and reduce the number of domains that need processing.** The interaction between them explains the method's performance better than the hardware label alone.

## Account for the cost of a proof

Consider a bounded continuous nonlinear program:

$$
p^\star = \min_{x\in B_0} f_0(x)
\quad\text{subject to}\quad
f_i(x)\leq 0,\qquad i=1,\ldots,m.
$$

A feasible point supplies an upper bound. A global solver must also produce lower bounds covering every remaining region. Spatial branch-and-bound does this by splitting the initial box, bounding the children, and pruning regions that are infeasible or cannot improve the incumbent.

For a run with approximately constant batch size $K$, a useful runtime accounting model is

$$
T_{\mathrm{search}}\approx
N_{\mathrm{visited}}\frac{T_{\mathrm{batch}}}{K}
+T_{\mathrm{queue}}.
$$

The first factor measures how much search remains. The second measures the effective cost of processing a domain, including bounding and primal recovery. Queue management adds overhead. A faster bound helps most when the rest of the processing can sustain its throughput and when that bound removes useful portions of the search space.

GEAR's design connects these requirements. Its bounds are tensor computations that also carry enough constraint geometry to improve pruning, primal initialization, and branching.

## Batch the nodes and the constraint outputs together

GEAR represents the objective and all constraints as outputs of one computation graph. Linear bound propagation traverses this graph backward through valid operator envelopes. On each sub-box $B$, it produces affine lower bounds

$$
\ell_i(x)=a_i^\top x+b_i\leq f_i(x),\qquad x\in B.
$$

For $K$ subdomains, $n$ variables, and $m$ constraints, the output is organized as

$$
A\in\mathbb{R}^{K\times(m+1)\times n},\qquad
b\in\mathbb{R}^{K\times(m+1)}.
$$

The batch dimension represents different regions of the same optimization problem. The output dimension represents the objective and constraints. Every region has its own coefficients and interval bounds, while all regions share the graph's computational structure. This makes a frontier of heterogeneous boxes suitable for the same sequence of batched operations.

The arithmetic grows with the batch, but exposes parallel work to the GPU. Downstream operations consume slices and combinations of these tensors, avoiding the construction and solution of a separate LP relaxation at each node.

## Turn one propagation pass into several kinds of progress

An affine bound contains more information than its scalar minimum over a box. Its coefficients describe which directions matter and how a constraint cuts across the domain. GEAR retains that information for the rest of the node processing.

For a feasible point, $f_i(x)\leq 0$ and $\ell_i(x)\leq f_i(x)$ imply $\ell_i(x)\leq 0$. Constraint lower bounds therefore become necessary linear conditions for feasibility. They can clip the input box and support infeasibility tests before more search is spent on it.

The same rows strengthen the objective bound. For nonnegative weights $\lambda$, form

$$
c(\lambda)=a_0+\sum_{i=1}^m\lambda_i a_i,
\qquad
\beta(\lambda)=b_0+\sum_{i=1}^m\lambda_i b_i.
$$

At every feasible point,

$$
c(\lambda)^\top x+\beta(\lambda)
\leq f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)
\leq f_0(x).
$$

The box case makes the computational advantage explicit. For $B=[l,u]$, the mixed row has the lower bound

$$
d_B(\lambda)=\beta(\lambda)
+\sum_{j=1}^n\min\{c_j(\lambda)l_j,\;c_j(\lambda)u_j\}.
$$

Mixing rows and minimizing the affine function over a box use tensor contractions and coordinate reductions. During the multiplier optimization, the propagated rows stay fixed. Repeated updates to $\lambda$ can reuse the expensive graph bounds, and constraint-aware clipping can further tighten the relaxation.

Validity holds for every nonnegative $\lambda$. A short tightening loop can return its best valid bound even if it stops well before convergence. This lets the solver spend a limited amount of computation improving a certificate whose validity is already established.

## Reuse geometry without growing the node state with depth

A parent constraint row remains valid on its child boxes because they lie inside the domain where the row was derived. GEAR forwards those rows to the children, giving the next batch useful geometry before it has computed fresh bounds.

Once a child runs its propagation pass, locally derived rows replace the inherited rows. The active row set stays bounded with respect to search depth. The representation carries information forward without requiring every descendant to retain an expanding history of cuts.

The optimized weights also inform branching. GEAR scores coordinate $j$ using the width of its interval and the magnitudes of the objective and constraint coefficients:

$$
s_j=(u_j-l_j)\left(|a_{0,j}|+
\sum_{i=1}^m\lambda_i^\star|a_{i,j}|\right).
$$

This connects split selection to information already obtained while tightening the lower bound. The score uses absolute contributions, so opposing coefficients do not hide a coordinate's influence through cancellation. A longest-edge fairness rule keeps the search from repeatedly splitting an already narrow coordinate while other dimensions remain wide. Computing the score is another reduction on cached rows.

## Keep primal recovery in the same batching regime

The effective node cost includes finding feasible points. If primal recovery becomes a sequence of expensive per-node subproblems, faster bounding alone cannot sustain the pipeline's throughput.

GEAR uses a projected augmented Lagrangian method whose inner steps are first-order updates on the original computation graph. The relaxation minimizer seeds the primal search; projected updates and outer multiplier updates batch across active domains. This puts primal recovery in the same computational regime as dual tightening and clipping.

The primal search uses its own multipliers $\mu$; the dual weights $\lambda$ mix affine relaxations. A candidate updates the incumbent only after a direct feasibility check against the original nonlinear constraints. The shared representation couples the searches while preserving the separate roles of a valid lower bound and a feasible upper bound.

## Measure both coverage and the amount of search

At a 180-second cap, the controlled comparison on 505 bounded continuous GAMS and MINLPLib instances reports 399 solves for GEAR and 334 for SCIP-Nonlinear. GEAR contributes 104 instances on which SCIP reaches the time limit; SCIP contributes 39 instances, and the combined coverage is 438. GEAR uses the CPU and GPU, and SCIP uses the CPU. [Paper, Section 4 and Appendix C](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

The cumulative ablation gives a more direct view of how the representation affects search. **All rows below use an 80-second cap**, distinct from the headline comparison:

| Configuration | Solved / 505 | Average visited domains |
| --- | ---: | ---: |
| Basic linear bound propagation + ALM-PGD | 248 | 7.94 million |
| Add relaxed clipping | 294 | 3.81 million |
| Add constraint-aware objective bounds | 360 | 1.58 million |
| Add complete clipping | 361 | 1.58 million |
| Add refined operator envelopes | 377 | 1.17 million |
| Add constraint-weighted branching | 381 | 1.04 million |
| Add ALM with linear constraints: full GEAR | 383 | 1.30 million |

The full configuration solves 135 more instances while reducing the average visited-domain count to about one-sixth of the basic configuration's count. Constraint-aware bounds and clipping account for much of the reduction; later components extend coverage further. [Paper, Table 2](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

The last row also shows why node count alone is insufficient: full GEAR visits more domains than the preceding branching configuration and solves more problems. Under a fixed budget, useful primal and dual work can justify additional search. These measurements describe the cumulative configurations; they do not isolate each component's GPU throughput.

## Choose a relaxation that can sustain the workload

GEAR propagates a single affine lower or upper bound for each nonlinear output in each backward step. For bilinear products, selecting a stronger valid McCormick plane improves the information that continues through the graph. Guarded fused quotient bounds avoid slack introduced by decomposing division into separate operations. These local choices improve the bounds used for clipping, dual tightening, and search guidance while keeping the backward pass tensorized.

A full LP relaxation can retain several facets simultaneously. That can close some problems at the root, where additional parallel search has little work to exploit. The paper's SCIP-only diagnostics include 23 of 39 instances solved at the root. GEAR gains when its regular coefficient computations and repeated reuse of bounds can process difficult search effectively. The governing tradeoff is **relaxation strength per node and the throughput at which useful nodes can be processed**. [Paper, Section 3.5 and Appendix C](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

Memory shapes this regime too. The dense coefficient tensor alone requires storage proportional to $K(m+1)n$, with intermediate graph bounds and clipping state adding to the budget. Increasing batch size exposes more parallel work while consuming more memory. Keeping the active constraint rows bounded with depth addresses one source of growth; variable count, constraint count, and graph structure still determine the batch sizes that fit.

GEAR's evaluated scope is bounded continuous NLPs over supported factorable operators. Within that scope, scalability depends on how well the solver can preserve useful constraint geometry, keep the node operations batched, and turn tighter bounds into less remaining proof work.
