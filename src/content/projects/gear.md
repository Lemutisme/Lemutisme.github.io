---
title: "GEAR"
tagline: "GPU-accelerated global nonlinear optimization through linear bound propagation."
description: "GEAR reuses propagated linear bounds to couple constraint relaxations, dual bounds, feasible solution search, and branching, solving 399 of 505 bounded continuous NLP instances within 180 seconds."
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

## Research question

Can the GPU throughput of neural network verification support a global solver for constrained nonlinear programs?

A global solver needs both feasible solutions and valid lower bounds on the best attainable objective. Nonlinear constraints must influence the bounds, the search domain, and the branching decisions. GEAR brings these operations together around the linear coefficients produced by bound propagation.

## One backward pass, multiple solver operations

GEAR represents the objective and all constraints as outputs of a single computation graph. A batched backward pass produces affine lower bounds across many branch-and-bound subdomains. The solver then reuses those coefficients throughout the search:

| Operation | How the propagated bounds are used |
| --- | --- |
| Constraint relaxation and clipping | Convert nonlinear constraints into necessary linear inequalities and reduce the search domain. |
| Dual lower bounding | Optimize nonnegative combinations of objective and constraint rows to strengthen the objective bound. |
| Infeasibility detection | Use the relaxations to certify that a subdomain cannot contain a feasible solution. |
| Primal recovery | Seed a projected augmented Lagrangian search for feasible solutions. |
| Branching | Use constraint-aware coefficient information to guide the next split. |

These operations run as batched tensor computations on the GPU. Tighter envelopes for bilinear products, quotients, and other NLP operators improve the bounds that feed the entire pipeline.

Primal candidates update the incumbent only after checks against the original nonlinear constraints. Valid lower bounds support pruning and optimality certification; primal recovery and branching guide the search.

## Benchmark results

The paper evaluates 505 bounded continuous NLP instances from GAMS and MINLPLib. In the controlled paired comparison, both solvers use the same workstation, a 180-second wall-clock budget, and the same optimality tolerance. GEAR uses the CPU and GPU; SCIP-Nonlinear uses the CPU.

| Result | GEAR | SCIP-Nonlinear |
| --- | ---: | ---: |
| Solved instances | 399 / 505 | 334 / 505 |
| Instances solved exclusively | 104 | 39 |

The two solvers solve 295 instances in common and 438 in total. GEAR adds coverage on large, highly nonlinear problems, while SCIP retains complementary strengths. These figures describe the paper's controlled comparison, rather than a universal speedup claim.

## Scope and connection to verification

GEAR targets bounded continuous nonlinear programs expressed using supported factorable operators. Integer variables and unbounded domains are outside the evaluated scope.

The project builds on the linear bound propagation machinery used in neural network verification and connects naturally to [our work on scalable verification](/projects/neural-network-verification/), including Clip-and-Verify. The [GEAR research note](/blog/gear/) develops the constraint-aware bound and explains how the primal and dual procedures fit into global search.
