---
title: "ProContract"
tagline: "The model sets the ceiling. Keeping the task intact sets the floor."
description: "A progress-contract kernel that holds a long-horizon agent's responsibilities outside the harness. The pass-rate gains are this kernel against the same model in the harness it starts from. A later self-written harness is optional, and it is not what produces those gains."
status: "Under review"
period: "2026"
citationYear: 2027
order: 3
tags:
  - Agentic AI
  - Long-Horizon Agents
  - Recursive Self-Improvement
  - AI Safety
links:
  - label: "Prototype"
    url: "https://github.com/Lemutisme/opencode/tree/capability-rsi"
pending:
  - arXiv coming soon
  - Code coming soon
bibtex: |
  @misc{zhou2027procontract,
    author    = {Zhou, Duo},
    title     = {{ProContract}: Decoupling Responsibility and Progress from the Harness in Long-Horizon Agents},
    year      = {2027},
    note      = {Under review},
    url       = {https://www.duo-zhou.com/projects/procontract/}
  }
---

**ProContract: Decoupling Responsibility and Progress from the Harness in Long-Horizon Agents**

Under review. arXiv coming soon · Code coming soon · [Prototype](https://github.com/Lemutisme/opencode/tree/capability-rsi)

## A long run fails by losing the task

A long-horizon agent already has a strong model. The failures that remain are about the record of the work. A requirement drops out while the agent delegates, compacts, or restarts. A check the agent ran on itself gets remembered as completion. Both happen for the same reason: the responsibility and the progress live in the execution state the harness keeps rewriting.

<figure>
  <img src="/figures/procontract/pc-fig1-architecture.png" alt="Three panels. A typical harness loses a requirement from its context and treats its own check as done. ProContract keeps contracts in a kernel the harness can only read, and records progress only when a verifier judges a submitted artifact. Improving the harness is itself a task: one version writes the next, and adoption stays outside the agent.">
  <figcaption>Responsibility and progress move out of the harness. The running agent reads the contract and submits artifacts. Only an authorized judgment writes progress. Panel (c) is a later option: the harness can be rewritten because it no longer holds the task. The pass rates below do not use that rewritten harness.</figcaption>
</figure>

The kernel is a small deterministic program. It runs no model. A contract stores one task's requirements, deadline, and budget, and only the user can change the requirements. The harness can forget a requirement in its own context. It cannot delete the requirement, because every acceptance is judged against the copy the kernel holds, and the task stays open until that judgment lands. A delegated subtask gets its own contract. The parent may rely on the subtask's acceptance, and may not treat the subagent's own report as progress.

## The same model, a higher floor

The comparison is the base kernel: ProContract before any self-improvement, running the instructions of the version self-improvement would later start from, against the public mini-SWE-agent run of the same model. No successor harness is involved.

On 200 ProgramBench tasks with GPT-5.6 Terra, that public run averages 72.35%. The base kernel averages 77.24%. On the 197 tasks both score, the paired gain is 5.33 points, with a 95% interval from 2.90 to 7.95. The gain is concentrated where the public run collapses: 23 tasks score below 50% instead of 39, and 5 score below 10% instead of 10. At the top the two runs meet. Twenty-eight tasks pass at least 95%, against 30.

<figure>
  <img src="/figures/procontract/pc-fig2-floor.png" alt="Pass-rate histograms and survival curves for mini-SWE-agent, ProContract base, and a self-written harness on 200 ProgramBench tasks. The public run has many more tasks below 50 percent. The curves meet again above 95 percent.">
  <figcaption>Hidden-test pass rates with GPT-5.6 Terra. Green is the base kernel, the system in the comparison above. Blue is a later self-written harness, drawn on the same axes and not the source of the lift. The public run's left tail shrinks under the base kernel. Above 95% the curves meet: the model sets how many tasks are essentially solved.</figcaption>
</figure>

Those floor tasks are tasks the model can already do. Eight of the ten public-run collapses score exactly zero. On five of the six that are not broadly hard for every system, the kernel run scores between 76% and 93%. The contract did not teach the model the task. It kept the task in front of a model that could finish it.

The kernel used for that gain treats a self-check as evidence, not as completion. A later run of a self-written harness under the same kernel makes the rule countable: all 420 passing self-checks left their tasks open, and 145 of the 199 tasks that had one went on to deliver a later artifact. That count is an illustration of the rule. It is not the run that produced the 5.33 points. Scored against the first artifact whose own check passed, the artifact finally delivered in that later run is better on 65 of 140 scored tasks and worse on 14. Delivery is the accepted artifact. In that later run, all 198 deliveries have the submission, the verifier's report, and the exported archive naming the same object. Three tasks that resumed inside a restarted harness kept their requirements, deadline, and delivery.

Holding the task open has a price. The base run costs $13.50 per task against $5.38 for mini-SWE-agent, because it keeps working and still carries its history in context. A fifth of its requests cross the long-context price tier. The contract is what makes that history optional.

## The harness can change because it does not own the task

The floor above is already there before any self-written harness. Self-improvement is what the separation allows afterwards, and the generated harness did not produce the gain.

Once responsibility and progress sit outside the harness, rewriting the harness is an ordinary task. Version 0 writes version 1. Version 1, running its own code, writes version 2. Both edits do one thing: they move evidence out of the context window and teach the agent to work from the stored record. That is the kernel's rule, applied by the agent to its own working memory. The edit is safe because compaction can no longer drop the responsibility. Without the kernel, the context is the task, and whatever compaction drops is gone.

Exploration and adoption are different decisions, and the kernel keeps the second one outside the version being judged. On the six development tasks that choose the next research parent, version 2 gains 6.37 points over version 0. On the three confirmation tasks that decide adoption, it loses 0.23 points. Version 0 stays the incumbent. Version 2 becomes the next parent and does not become the deployed harness.

Unadopted, it can still be run, because a generated harness can request and cannot write the contract. It ran all 200 Terra tasks with every contract intact. Against the base kernel run it keeps the floor at about half the cost: $7.22 per task instead of $13.50, and requests above 272K input tokens fall from 8,481 to 898. Tasks below 10% go from 5 to 4. The mean on tasks both score falls by 2.16 points, and the loss sits in the quarter whose context shrank the most. The context had been storing capability. The contract was storing the task.

## A second model, and tasks from after the model's cutoff

On the official ProgramBench leaderboard every published entry uses mini-SWE-agent. GPT-5.6 Sol at xhigh averages 69.89% there. ProContract with the same model and effort averages 75.25%, a paired gain of 5.36 points (95% interval 2.46 to 8.39). The shape repeats: 3 tasks below 10% instead of 15, and 30 tasks at or above 95% instead of 31. Across the plotted leaderboard this is the highest mean pass rate. The same model resolves five tasks instead of two. The only entry that resolves more, Claude Opus 5 at xhigh, costs about sixteen times as much per task.

<figure>
  <img src="/figures/procontract/pc-fig3-leaderboard.png" alt="ProgramBench leaderboard plots of resolved rate and mean pass rate against cost. ProContract with GPT-5.6 Sol sits above the official mini-SWE-agent frontier on mean pass rate, at similar cost to that model's own official run.">
  <figcaption>Official ProgramBench entries, all mini-SWE-agent, and two ProContract runs. The arrow is the same model, GPT-5.6 Sol at xhigh, with and without the kernel: mean pass rate moves by 5.4 points. Resolved rate moves less, and a much more expensive run still leads that axis.</figcaption>
</figure>

Terminal-Bench 2.0 was released after GPT-5.2's published knowledge cutoff, and that model has the most public harnesses on the leaderboard. ProContract passes 65.7% ± 1.1 of the 89 tasks over two trials. The public GPT-5.2 harnesses sit at 64.9% (Droid), 62.9% (Codex CLI), 60.7% (Mux), and 54.0% (Terminus 2). This particular run includes the compaction edit from the self-written harness. The ProgramBench gains above do not. Those are the base kernel against mini-SWE-agent.

<figure>
  <img src="/figures/procontract/pc-fig4-terminal.png" alt="Terminal-Bench 2.0 pass rates with GPT-5.2. ProContract at 65.7 percent, then Droid 64.9, Codex CLI 62.9, Mux 60.7, and Terminus 2 at 54.0, with interval bars.">
  <figcaption>Terminal-Bench 2.0, all 89 tasks, GPT-5.2. Gray points are public harnesses for that model. The interval on ProContract is two trials.</figcaption>
</figure>

## What the kernel does not decide

The kernel decides who may record progress. It does not decide whether the specification is complete or the verifier is right, so a stronger harness can no longer hide a weak judge. The self-improvement run is controlled self-modification: a generated version rewrote its own harness and could not adopt itself. The prototype withdraws support at the level of a whole contract, which reopens more tasks than a finer record would. That is the conservative side of the design.
