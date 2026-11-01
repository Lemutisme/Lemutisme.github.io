---
title: "Scalable Neural Network Verification"
tagline: "Optimization and search methods that turn formal guarantees into practical verification systems."
description: "A research program on stronger relaxations, tighter domain reduction, and more informed branching for complete neural network verification."
status: "Active research"
period: "2024 - Present"
citationYear: 2026
order: 4
tags:
  - Formal Methods
  - Neural Networks
  - Optimization
  - Branch and Bound
links:
  - label: "BICCOS - NeurIPS 2024"
    url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html"
  - label: "Clip-and-Verify - NeurIPS 2025"
    url: "https://neurips.cc/virtual/2025/poster/118862"
  - label: "Lookahead Branching - IJCAI 2026"
    url: "https://openreview.net/forum?id=lX9Eto4L1O"
  - label: "BICCOS code"
    url: "https://github.com/Lemutisme/BICCOS"
  - label: "Clip-and-Verify code"
    url: "https://github.com/Verified-Intelligence/Clip_and_Verify"
bibtex: |
  @inproceedings{zhou2024biccos,
    author    = {Zhou, Duo and Brix, Christopher and Hanasusanto, Grani A. and Zhang, Huan},
    title     = {Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes},
    booktitle = {Advances in Neural Information Processing Systems},
    volume    = {37},
    pages     = {29324--29353},
    year      = {2024}
  }

  @inproceedings{zhou2025clipandverify,
    author    = {Zhou, Duo and Chavez, Jorge and Chen, Hesun and Hanasusanto, Grani A. and Zhang, Huan},
    title     = {Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification},
    booktitle = {Advances in Neural Information Processing Systems},
    volume    = {38},
    pages     = {174849--174895},
    year      = {2025}
  }

  @inproceedings{davis2026lookahead,
    author    = {Davis, Liam and Zhou, Duo and Zhang, Huan and Katz, Guy and Barrett, Clark W. and Wu, Haoze},
    title     = {Lookahead Branching for Neural Network Verification},
    booktitle = {Proceedings of the 35th International Joint Conference on Artificial Intelligence},
    year      = {2026}
  }
---

## Research question

Complete neural network verification can certify that every input in a specified region satisfies a safety property. The challenge is not the definition of the guarantee, but scaling the proof search to realistic networks and difficult specifications.

This project studies a common bottleneck: a branch-and-bound verifier repeatedly spends computation on domains whose relaxations are too weak or whose branching choices reveal too little about the remaining search space.

## A progression of methods

### BICCOS: learning useful cutting planes from search

[BICCOS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html) extracts cutting planes from branch-and-bound conflicts. Instead of treating explored branches as discarded computation, it converts them into globally useful constraints that strengthen later relaxations.

The central perspective is that the search tree is also a source of proof structure. A scalable verifier should learn from infeasible or dominated subproblems rather than repeatedly rediscovering the same geometry.

A cut read off one infeasible path is often already implied by every other open subproblem, so recording it does not move the bound. BICCOS checks the reduced path and, when the cut is not redundant, shares it across the other branches that are being searched at the same time.

<figure>
  <img src="/figures/neural-network-verification/biccos-cuts.png" alt="Diagram of branch-and-bound cuts. One panel shows a cut inferred from an unsatisfiable path that does not improve bounds because other subproblems already satisfy it. Another shows a reduced path that is still verifiable and yields a new cut. A third shows that cut shared across subtrees.">
  <figcaption>An infeasible path is a candidate cut, not yet a useful one. The cut is kept when the path with the redundant splits removed is still infeasible, and it is then applied to the other subtrees rather than only to the branch that found it.</figcaption>
</figure>

### Clip-and-Verify: reduce the domain before expensive search

[Clip-and-Verify](https://neurips.cc/virtual/2025/poster/118862) uses linear constraints to clip away regions that cannot contain a counterexample. The verifier first invests in tightening the feasible domain, then runs complete verification on the smaller residual problem.

This changes where computation is spent: remove irrelevant volume early so that later bounding and branching operate on a more informative domain. The linear constraints that do the clipping are already produced by bound propagation. Split constraints and output constraints both cut the input box, and the cut is applied with a GPU procedure instead of a call to an external LP solver. On some VNN-COMP instances the number of branch-and-bound subproblems falls by 96%.

<figure>
  <img src="/figures/neural-network-verification/clip-idea.png" alt="Diagram of domain clipping. Linear bounds and split or output constraints cut the input box. Relaxed clipping removes part of the box. Complete clipping removes the entire region on the infeasible side of the constraint, which tightens a neuron bound.">
  <figcaption>Clipping uses constraints the verifier already has. Relaxed clipping removes a corner. Complete clipping removes everything on the wrong side of a constraint, and the neuron bound tightens because the remaining box is smaller.</figcaption>
</figure>

### Lookahead Branching: evaluate the consequence of a split

[Lookahead Branching](https://openreview.net/forum?id=lX9Eto4L1O) improves branching by estimating how candidate splits change downstream verification bounds. Rather than using only a local proxy score, it asks which branch is likely to produce the most useful proof progress. On α-β-CROWN and on Marabou, the lookahead curves sit under the standard branching heuristic: same instances, less time, and up to 57% more instances solved.

<figure>
  <img src="/figures/neural-network-verification/lookahead-cactus.png" alt="Six cactus plots comparing lookahead branching with FSB on alpha-beta-CROWN. Lookahead's curve is below FSB on CIFAR, TinyImageNet, and MNIST benchmarks.">
  <figcaption>α-β-CROWN. Each curve is instances sorted by the time to verify them. Lookahead is the lower curve: it spends its first splits on the branch that changes the bound, instead of on the locally largest score.</figcaption>
</figure>

Together, these methods attack three complementary dimensions of verification complexity:

1. strengthen the relaxation;
2. shrink the feasible domain; and
3. make better search decisions.

## Systems impact

The work is developed in the alpha-beta-CROWN ecosystem and targets end-to-end complete verification. The resulting verifier won both the Regular and Extended tracks at VNN-COMP in 2024, 2025, and 2026.

## Current directions

- Unifying domain clipping, cutting planes, and branching under a shared measure of proof progress.
- Making lookahead decisions cheaper through bound reuse and batched GPU computation.
- Extending complete verification to richer nonlinear and structured specifications.
- Connecting solver performance improvements to interpretable guarantees about search reduction.
