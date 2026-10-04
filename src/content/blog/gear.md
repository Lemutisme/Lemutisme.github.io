---
title: "GEAR: From Neural Network Verification to Global Nonlinear Optimization"
description: "How GEAR turns propagated linear coefficients into constraint relaxations, dual lower bounds, primal seeds, and branching decisions in a GPU-accelerated global NLP solver accepted to NeurIPS 2026."
publishedAt: 2026-10-03
tags:
  - Global Optimization
  - Nonlinear Programming
  - Linear Bound Propagation
  - GPU Computing
draft: false
---

Our paper, **GEAR: A GPU-Accelerated Global Solver for Nonlinear Programs via Linear Bound Propagation**, has been accepted to **NeurIPS 2026**.

**Duo Zhou**, Hesun Chen, Xiangru Zhong, Grani A. Hanasusanto, Huan Zhang

[Download the PDF](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing) · [OpenReview](https://openreview.net/forum?id=3WdBmyLLLC) · [Project page](/projects/gear/)

## From verifying a property to optimizing an objective

Neural network verification has developed an efficient way to bound the outputs of computation graphs over input regions. Linear bound propagation traverses the graph backward through valid operator relaxations, producing affine bounds whose coefficients can be computed in batches on GPUs.

GEAR asks how this machinery can support constrained global nonlinear optimization. Consider a bounded continuous nonlinear program:

$$
p^\star = \min_{x\in B_0} f_0(x)
\quad\text{subject to}\quad
f_i(x)\leq 0,\qquad i=1,\ldots,m.
$$

Finding a feasible point gives an upper bound on the optimal value. Certifying global optimality also requires a lower bound that covers every remaining search region. The constraints matter on both sides: they determine whether a candidate is feasible and which parts of the box can actually attain a low objective.

GEAR encodes the objective and constraints together as a vector-valued computation graph. One batched backward pass produces affine lower bounds for all outputs across the current branch-and-bound subdomains. The resulting coefficients become shared data for the solver's bounding, clipping, primal recovery, and branching operations.

## Keep the coefficients and use the constraints

On a sub-box $B$, let bound propagation return

$$
\ell_i(x)=a_i^\top x+b_i\leq f_i(x),\qquad x\in B.
$$

If $x$ satisfies the original constraint $f_i(x)\leq 0$, it must also satisfy $\ell_i(x)\leq 0$. Each constraint's affine lower bound therefore supplies a necessary linear condition for feasibility.

These inequalities can reduce the input domain through clipping and strengthen subsequent bounds. A linear inequality derived on a parent box remains valid on its descendants. Reusing it preserves information about the feasible region as search proceeds.

The coefficients also let us combine objective and constraint information. For any nonnegative weights $\lambda_i$, define

$$
\ell_\lambda(x)=\ell_0(x)+\sum_{i=1}^m\lambda_i\ell_i(x).
$$

At any feasible point,

$$
\ell_\lambda(x)
\leq f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)
\leq f_0(x).
$$

Minimizing this mixed affine row over an outer relaxation of the feasible set yields a valid objective lower bound. GEAR improves the bound by optimizing the nonnegative weights over the propagated rows. The weights are parameters of the relaxation; they need not be optimal multipliers of the original nonlinear program for the bound to be valid.

A simple example illustrates the effect. Minimize $x$ over $[0,2]$ subject to $1-x\leq 0$. Bounding the objective over the box alone gives zero. Combining the exact affine rows with $\lambda=1$ gives $x+(1-x)=1$, which certifies the constrained optimum. In a nonlinear problem, operator relaxations supply the affine rows that make the same reasoning possible.

## Make the entire node computation fit the GPU

For a batch of $K$ subdomains with $n$ variables and $m$ constraints, the backward pass produces coefficient tensors with shapes

$$
A\in\mathbb{R}^{K\times(m+1)\times n},\qquad
b\in\mathbb{R}^{K\times(m+1)}.
$$

The downstream solver operations consume these tensors directly. Constraint relaxation, lower bounding, infeasibility checks, and search guidance can therefore share one stream of coefficients across many nodes.

This design also makes operator bounds consequential. An NLP graph may contain products, quotients, powers, logarithms, reciprocals, and trigonometric terms. GEAR develops tighter envelopes for these operators, including coefficient-aware selection of McCormick planes for bilinear products and guarded fused relaxations for quotients. Improving these local bounds strengthens the information available throughout the solver.

## Recover feasible solutions and close the gap

GEAR couples the dual procedure to a projected augmented Lagrangian method for primal recovery. The minimizer associated with the linear bound seeds the primal search, and first-order updates run in batches across subdomains. A candidate becomes an incumbent only after direct feasibility checks against the original nonlinear constraints.

The branch-and-bound loop then combines both kinds of progress:

1. Valid lower bounds and infeasibility certificates remove subdomains that cannot improve the incumbent or contain a feasible point.
2. Feasible primal candidates improve the global upper bound.
3. Constraint-aware branching selects how to split the surviving boxes.

The bound computations support certification; primal recovery and branching determine how efficiently the search reaches it. The paper states the soundness conditions and the assumptions needed for finite tolerance-based termination.

## What the controlled benchmark shows

The evaluation uses 505 bounded continuous instances: 265 from `gams_world` and 240 from MINLPLib. The controlled GEAR-versus-SCIP comparison uses matching instances, the same optimality tolerance, and a 180-second wall-clock limit on the same workstation. GEAR uses the CPU and an NVIDIA RTX 5090 GPU; SCIP-Nonlinear uses the CPU. [Paper, Section 4](https://drive.google.com/file/d/1BFk00R0f4ARasqnQLLde5q1xA_Ssf41c/view?usp=sharing).

| Coverage at 180 seconds | Instances |
| --- | ---: |
| Solved by GEAR | 399 / 505 |
| Solved by SCIP-Nonlinear | 334 / 505 |
| Solved by both | 295 |
| Solved only by GEAR | 104 |
| Solved only by SCIP-Nonlinear | 39 |
| Solved by either solver | 438 / 505 |

GEAR's exclusive solves are concentrated on large, highly nonlinear problems. The overlap also shows why solver portfolios are useful: each method solves instances the other misses. The paper reports broader comparisons with commercial and open-source solvers, with separate experimental settings; the paired table above is the controlled comparison.

GEAR targets bounded continuous programs over supported factorable operators. Its results establish a useful GPU regime for this class of global optimization problems. They also extend a line of work from [scalable neural network verification](/projects/neural-network-verification/) to a solver that uses the objective and nonlinear constraints together throughout global search.
