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

**BICCOS (NeurIPS 2024) · Clip-and-Verify (NeurIPS 2025) · Lookahead Branching (IJCAI 2026)**

Winner of the Regular and Extended tracks at **VNN-COMP 2024, 2025, and 2026**.

## Three levers on branch-and-bound complexity

Complete neural network verification certifies that every input in a specified region satisfies a safety property. Because exact certification over nonlinear activations is NP-hard, modern verifiers combine linear bound propagation with spatial or activation branch-and-bound (BaB).

When a verifier times out on a realistic network, the bottleneck sits in one of three places: the per-node convex relaxation is too loose to prune, the input box still contains large regions that already violate linear constraints, or the branching heuristic splits neurons that barely move the lower bound. This research program attacks those three axes inside the $\alpha,\beta$-CROWN ecosystem:

| Bottleneck in branch-and-bound | Mechanism | Effect on proof search |
| --- | --- | --- |
| Weak relaxations re-explore the same conflicts | [BICCOS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html) (NeurIPS 2024) | Extracts reduced conflict cores from verified subtrees and shares them as global cutting planes. |
| Expensive bounding runs over irrelevant box volume | [Clip-and-Verify](https://neurips.cc/virtual/2025/poster/118862) (NeurIPS 2025) | Uses existing linear bounds to clip the input domain on the GPU before further splitting. |
| Local branching scores mispredict downstream progress | [Lookahead Branching](https://openreview.net/forum?id=lX9Eto4L1O) (IJCAI 2026) | Evaluates how candidate splits tighten multi-step downstream bounds. |

## BICCOS: turn dead-end branches into global cutting planes

A standard branch-and-bound verifier treats a verified or infeasible leaf as finished work and discards it. [BICCOS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html) treats that leaf as a proof certificate about which activation patterns cannot coexist.

The technical obstacle is that a constraint read directly off a full root-to-leaf path only excludes that exact leaf, which is already closed: every other open subproblem satisfies it trivially, so adding it as a cut does not move any bound. BICCOS shrinks the path to a minimal conflict core by dropping splits whose removal leaves the subproblem verifiable. That shorter clause rules out an entire combination of neuron states and strengthens the relaxation across all remaining sibling subtrees.

<figure>
  <img src="/figures/neural-network-verification/biccos-cuts.png" alt="Diagram of branch-and-bound cuts. One panel shows a cut inferred from an unsatisfiable path that does not improve bounds because other subproblems already satisfy it. Another shows a reduced path that is still verifiable and yields a new cut. A third shows that cut shared across subtrees.">
  <figcaption>An infeasible path is a candidate cut, not yet a useful one. The cut is kept when the path with the redundant splits removed is still infeasible, and it is then applied to the other subtrees rather than only to the branch that found it.</figcaption>
</figure>

## Clip-and-Verify: remove infeasible volume before splitting

Bound propagation produces linear inequalities over the input box at every node, yet standard verifiers use those inequalities only to compute a scalar lower bound at the output. [Clip-and-Verify](https://neurips.cc/virtual/2025/poster/118862) turns those same linear constraints back onto the input domain to shave off regions that cannot contain a counterexample.

Relaxed clipping trims coordinate intervals directly; complete clipping solves the linear program over the box and the active linear constraints to contract each input dimension before the next propagation pass. Because maximizing or minimizing a linear function over a hyperrectangle admits a closed-form coordinate reduction in the Lagrangian dual, complete clipping reduces to a low-dimensional dual update that batches natively on the GPU without calling an external CPU LP solver. The tighter input box immediately shrinks intermediate pre-activation intervals and reduces branch-and-bound subproblems on hard VNN-COMP instances by up to 96%.

<figure>
  <img src="/figures/neural-network-verification/clip-idea.png" alt="Diagram of domain clipping. Linear bounds and split or output constraints cut the input box. Relaxed clipping removes part of the box. Complete clipping removes the entire region on the infeasible side of the constraint, which tightens a neuron bound.">
  <figcaption>Clipping uses constraints the verifier already has. Relaxed clipping removes a corner. Complete clipping removes everything on the wrong side of a constraint, and the neuron bound tightens because the remaining box is smaller.</figcaption>
</figure>

## Lookahead Branching: split for downstream bound movement

Standard branching heuristics score a candidate neuron by its single-step bound change. In deep networks, a split that moves the bound slightly at step one can unlock a cascade of stable neurons two steps later, while a locally large split stalls. [Lookahead Branching](https://openreview.net/forum?id=lX9Eto4L1O) estimates multi-step downstream bound tightening before committing to a split. Across both $\alpha,\beta$-CROWN and Marabou, the lookahead curves sit below the standard Filtered Smart Branching (FSB) heuristic: less verification time per instance, and up to 57% more instances certified within the timeout.

<figure>
  <img src="/figures/neural-network-verification/lookahead-cactus.png" alt="Six cactus plots comparing lookahead branching with FSB on alpha-beta-CROWN. Lookahead's curve is below FSB on CIFAR, TinyImageNet, and MNIST benchmarks.">
  <figcaption>α-β-CROWN. Each curve is instances sorted by the time to verify them. Lookahead is the lower curve: it spends its first splits on the branch that changes the bound, instead of on the locally largest score.</figcaption>
</figure>

## From neural verifiers to general nonlinear solvers

These methods ship in the open-source $\alpha,\beta$-CROWN verifier, which won both the Regular and Extended tracks at the International Verification of Neural Networks Competition (VNN-COMP) in 2024, 2025, and 2026.

The deeper takeaway of this line of work is that GPU-batched linear bound propagation and constraint-driven domain clipping are not restricted to ReLU networks. Any factorable nonlinear program or discretized physical system is a computation graph. [GEAR](/projects/gear/) extends this machinery into a general-purpose GPU global solver for nonlinear programming, and a companion [research note](/blog/global-optimality-certification-pde/) studies what happens when discretized spatiotemporal PDEs are compiled into verification graphs via exact equality elimination.
