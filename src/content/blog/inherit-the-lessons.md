---
title: "Inherit the Lessons, Not the Harness"
description: "Without a ground-truth verifier, same-task contrastive reruns can justify which harness edits to keep (Harness², Xu et al., 2026); the transferable object across tasks is the evidence-backed editing lesson in the improvement library rather than the task-customized execution harness itself, so each new task still starts from a shared baseline."
publishedAt: 2026-11-10
tags:
  - Agent Harnesses
  - Recursive Self-Improvement
  - LLM Agents
  - Test-Time Adaptation
draft: false
---

**Primary Source (Paper-Reading Note):** **Harness²: Recursive Agent Harnessing for an Open World** · **arXiv:2610.14224 (2026)**

Ruiyao Xu (Northwestern University; work done while interning at Google Cloud AI Research), Yanfei Chen, Zhongying CuiZhu, Bhavana Dalvi Mishra, Yifei Ming, Han Yu, Rujun Han, Chen-Yu Lee, Tomas Pfister (Google Cloud AI Research)

[arXiv](https://arxiv.org/abs/2610.14224) · [PDF](https://arxiv.org/pdf/2610.14224) · [Code](https://github.com/google-research/harness2) · [Project site](https://harness2.github.io)

*Attribution note: This article is a paper-reading note on **Harness²: Recursive Agent Harnessing for an Open World** by **Ruiyao Xu, Yanfei Chen, Zhongying CuiZhu, Bhavana Dalvi Mishra, Yifei Ming, Han Yu, Rujun Han, Chen-Yu Lee, and Tomas Pfister** ([arXiv:2610.14224](https://arxiv.org/abs/2610.14224))—that paper is **not Duo Zhou's own work**, and all `Harness²` problem formulations, mechanisms, tables, figures, and empirical findings cited below belong to its authors.*

When a frozen language model $\mathcal{M}$ acts as an autonomous agent, its behavior is shaped by its execution harness $\mathcal{H} = (c_1, \dots, c_K)$: the system prompt, guardrails, memory files, tool guidance, subagent specifications, on-demand skills, helper scripts, and middleware hooks that govern what enters context and how tools execute ([Xu et al., 2026, Section 2.1, Figure 2, and Appendix D.3, Table 5](https://arxiv.org/abs/2610.14224)). Evolving $\mathcal{H}$ at test time is appealing because edits are file-level diffs around a frozen backbone rather than weight updates. Yet in an open-world stream of novel tasks $q_1, \dots, q_N$ without ground-truth labels, unit-test graders, or validation rubrics, **arbitrary harness edits easily make the agent worse**. In fact, when Xu et al. (2026) let an editor directly rewrite the base execution harness for each task without probing executions or consulting prior editing experience, scores on `WorkBuddy-Bench` with `Gemini 3.5 Flash` under `OpenCode` drop from $73.4$ to $63.2$ ($-10.2$ points) on `WB-code` and from $69.9$ to $65.0$ ($-4.9$ points) on `WB-web` ([Section 4, Figure 7 left](https://arxiv.org/abs/2610.14224)).

The core insight of **Harness²** ([Xu et al., 2026](https://arxiv.org/abs/2610.14224)) is that open-world harness evolution must separate **how an edit is justified on the current task** from **what object is carried forward to the next task**:

> **Without a ground-truth verifier, same-task contrastive reruns can justify which harness edits to keep; the transferable object across tasks is the evidence-backed editing lesson in the improvement harness, not the task-customized execution harness itself, so each new task still starts from a shared baseline $\mathcal{H}_0$.**

Reading the paper cleanly requires keeping those two claims distinct: (a) **same-task improvement** comes from probing candidate harness edits on $q_t$ under matched conditions and composing only changes backed by trajectory contrasts; (b) **cross-task transfer** comes from distilling those contrasts into an improvement library $\mathcal{R}_t$ while resetting the next task $q_{t+1}$ to the clean baseline harness $\mathcal{H}_0$.

<figure>
  <img src="/blog/inherit-the-lessons/contrastive-reruns-vs-lesson-transfer.svg" alt="Two-stage architecture diagram of Harness². Stage 1 shows task-level recursion on task q_t: starting from the shared baseline harness H_0, invariant (INV) and domain (DOM) candidate edits are executed on the same task under matched conditions and compared against reference rollout e_0 before selective composition into H*_t. Stage 2 contrasts naive cross-task inheritance of H*_t or unprobed direct editing (which lowers WB-code from 73.4 to 63.2) against harnessing-level recursion, which resets q_{t+1} to H_0 and transfers only contrast-backed lessons and adaptable components inside R_{t+1}.">
  <figcaption>The two coupled recursions in Harness² (Xu et al., 2026, Sections 2.2–2.3, Figure 3, and Figure 7). Task-level recursion uses matched same-task reruns to justify edits without a verifier; harnessing-level recursion transfers validated editing lessons in the updated improvement harness while resetting each new task to the baseline execution harness.</figcaption>
</figure>

## 1. Why harness edits are hard to justify without a verifier

Recursive self-improvement (RSI) asks an AI system to turn execution outcomes into persistent modifications that also shape how future improvements are generated and retained ([Xu et al., 2026, Section 1 and Section 2.1](https://arxiv.org/abs/2610.14224)). When the object of RSI is the agent harness $\mathcal{H} = (c_1, \dots, c_8)$ around a frozen model $\mathcal{M}$—forming an agent $\mathcal{A}(\cdot; \mathcal{M}, \mathcal{H})$—offline methods such as `Meta-Harness` (Lee et al., 2026b) or `AHE` (Lin et al., 2026) can search over candidate component trees against a fixed training split with ground-truth evaluators and deploy the winner ([Section 1, Section 3.1, and Appendix A, Table 4](https://arxiv.org/abs/2610.14224)).

Once the agent is deployed on an open-world stream $q_1, \dots, q_N$ of unseen coding, web, office, or legal workflows, that offline supervision disappears:

1. **No ground-truth verifier at test time.** Neither the executor nor the editor sees a grading rubric, unit-test verdict, or scalar score during deployment (enforced by a workspace guard in [Appendix D.2 and Appendix H, Prompt 1](https://arxiv.org/abs/2610.14224)). Unlike weight-space RLVR or offline harness search, an online harness editor has no external oracle telling it whether a rewrite of `systemprompt.md`, a new `skills/<n>/SKILL.md`, or a `plugin/<name>.ts` hook improved the final deliverable.
2. **Unverified harness edits have high downside risk.** In `Harness²`, the editable surface spans $K = 8$ file-level components ([Section 2.1, Figure 2, and Appendix D.3, Table 5](https://arxiv.org/abs/2610.14224)): four always-on context files (`C1` `systemprompt.md`, `C2` `guardrails.md`, `C3` `LongTermMEMORY.md`, `C4` `tool_descriptions/tool_guidance.md`), two on-demand context/orchestration components (`C5` `sub_agents/<n>/AGENT.md` via the `task` tool and `C6` `skills/<n>/SKILL.md` via the `skill` tool), deterministic bash helpers (`C7` `scripts/<name>.py`), and tool-loop middleware (`C8` `plugin/<name>.ts`). Every word added to `C1`–`C4` competes directly with the model's output budget on every call, and an ill-conceived rule or script can stall execution until the $200$-turn / $3{,}600$-second cap expires with no deliverable ([Appendix D.1 and Appendix H, Prompt 1](https://arxiv.org/abs/2610.14224)).
3. **Why naive carry-forward across tasks compounds noise.** If a system directly passes the task-$t$ modified execution harness forward to task $t+1$ (as in batch-carry methods; [Section 1](https://arxiv.org/abs/2610.14224)), task-specific assumptions—such as exact column mappings between two workspace spreadsheets or a task-specific position rule—become baked into the default runtime and misdirect subsequent tasks as requirements shift.

To evolve a harness online without labels, an RSI system needs a substitute for a verifier *within* task $q_t$, and a filter on *what* crosses the boundary between task $q_t$ and task $q_{t+1}$.

## 2. Same-task contrastive reruns as evidence

### 2.1 The Propose–Probe–Compose step

`Harness²` addresses the missing verifier at the task level by replacing scalar correctness feedback with **behavioral contrast across matched reruns on the same task** ([Section 2.2.1](https://arxiv.org/abs/2610.14224)). Let $e = (\tau, y)$ denote an execution record consisting of trajectory $\tau$ and deliverable $y$. Holding the frozen model $\mathcal{M}$, the environment, and the current improvement harness $\mathcal{R}_t$ fixed during task $q_t$, one contrastive improvement step applies an operator $\Phi$ to the current execution harness $\mathcal{H}$ and task-local history $\mathcal{E}$:

$$
(\mathcal{H}', e', \xi) = \Phi(q_t, \mathcal{H}, \mathcal{E}; \mathcal{R}_t).
$$

[Xu et al., 2026, Section 2.2.1, Equation (1)](https://arxiv.org/abs/2610.14224).

What the paper means when it says this step "reliably guides edits without ground-truth verifiers" is **not** that a rerun proves deliverable $y'$ is optimal. As Section 2.2.1 ("Probe") explicitly notes, *these contrasts inform subsequent composition rather than verify task correctness, as execution stochasticity and shared model biases limit the reliability of the evidence*. Instead, Equation (1) imposes a specific three-part evidentiary discipline ([Section 2.2.1 and Appendix H, Prompts 1–3](https://arxiv.org/abs/2610.14224)):

1. **Reference rollout and orthogonal scoping (`Propose`):** The agent first executes the unmodified harness to obtain a reference record $e_0 \sim \mathcal{A}(q_t; \mathcal{M}, \mathcal{H})$. Rather than asking one generic editor to rewrite everything, `Propose` drafts two orthogonal candidate harnesses from the same evidence:
   - An **invariant (`INV`) candidate** targeting cross-domain agentic execution failures: tool-call errors, unverified file writes, context overflow, or missing stop conditions ([Section 2.2.1 and Appendix H, Prompt 1](https://arxiv.org/abs/2610.14224)).
   - A **domain-specific (`DOM`) candidate** targeting practitioner requirements: artifact contracts, reconciliation hierarchies, required sections, and domain consistency checks ([Section 2.2.1 and Appendix H, Prompt 2](https://arxiv.org/abs/2610.14224)).
   Before any rollout occurs, a static validator rejects any candidate that violates three structural invariants: every `skills/<name>/SKILL.md` must have a verbatim trigger line in `systemprompt.md` or `guardrails.md`, always-on text cannot contain raw Jinja `{{` template braces, and every script referenced by a skill or prompt must ship on the editable surface (retried once on failure, otherwise falling back to the starting tree; [Appendix D.2 and Appendix H, Prompt 1](https://arxiv.org/abs/2610.14224)).
2. **Matched-condition execution (`Probe`):** Both candidate harnesses are executed on the *same* task $q_t$ in fresh containers under identical model and environment settings ([Section 2.2.1 and Appendix B](https://arxiv.org/abs/2610.14224)). Comparing the candidate trajectories $(\tau_{\mathrm{INV}}, \tau_{\mathrm{DOM}})$ against $\tau_0$ isolates what changed in the agent's behavior: e.g., whether a new `tool_guidance.md` stopped the model from emitting pseudo-tool calls as plain text (`WB-code`, $60.0 \to 100.0$; [Appendix E.7, Figure 15](https://arxiv.org/abs/2610.14224)), or whether a `verify_outputs.py` script and reconciliation skill fixed cross-file mismatches (`JobBench`, $43.1 \to 77.8$; [Section 4, Figure 9](https://arxiv.org/abs/2610.14224)).
3. **Selective, trajectory-anchored merging (`Compose`):** The composer inspects $e_0$, the candidate diffs against $\mathcal{H}$, and the probe trajectories. Every decision in its audit log $\xi$ must take the form `kept | revised | dropped -> <edit> -> <specific trajectory moment or base-set pattern>` ([Appendix H, Prompt 3](https://arxiv.org/abs/2610.14224)). Crucially, `Compose` is instructed to treat any edit associated with an empty or non-delivering rollout as unsafe, and to return `KEEP_BASE` whenever the probe trajectories do not show a concrete behavioral improvement.

### 2.2 Scaling task-level recursion: parallel vs. sequential

When additional test-time budget $k > 1$ is available on task $q_t$, `Harness²` scales $\Phi$ in two ways while keeping $\mathcal{R}_t$ fixed ([Section 2.2.2, Figure 4, Equations (2)–(3)](https://arxiv.org/abs/2610.14224)):

- **Parallel recursion (width $k$):** Collects $k$ reference executions $\mathcal{B}$ under $\mathcal{H}$, proposes and probes $k$ `INV` and $k$ `DOM` candidates to form a pool $\mathcal{P}$ of $2k$ probed candidates, and runs $k$ independent composition sessions over the full pool:

$$
\mathcal{H}'_j = \operatorname{Compose}_j(q_t, \mathcal{H}, \mathcal{B}, \mathcal{P}; \mathcal{R}_t), \qquad j = 1, \dots, k.
$$

- **Sequential recursion (depth $k$):** Starts from $\mathcal{H}^{(0)} = \mathcal{H}_0$ (with $k$ base executions in $\mathcal{E}^{(0)}$) and chains $k$ contrastive updates, passing the previous checkpoint's rollout as on-policy evidence into round $r$:

$$
(\mathcal{H}^{(r)}, e^{(r)}, \xi^{(r)}) = \Phi(q_t, \mathcal{H}^{(r-1)}, \mathcal{E}^{(r-1)}; \mathcal{R}_t), \qquad r = 1, \dots, k.
$$

For $k > 1$, a terminal **blind selector** ([Section 2.2.2 and Appendix H, Prompt 5](https://arxiv.org/abs/2610.14224)) compares the anonymized, shuffled deliverables against the task request and source materials—with no access to which harness produced which artifact, and with unmodified base rollouts included as valid choices—to select the final output and its corresponding adopted harness $\mathcal{H}^\star_t$. At $k = 1$, both modes reduce to a single contrastive step (one base rollout, one `INV` probe, one `DOM` probe, and one composed execution) and bypass selection ([Section 2.2.2](https://arxiv.org/abs/2610.14224)).

### 2.3 What the main benchmark results establish

Xu et al. (2026) evaluate four executor–harness configurations—frozen `Gemini 3.5 Flash` and `GLM-5` executors inside off-the-shelf `OpenCode` and `Codex` initial harnesses, with `Claude Opus 4.8` under `OpenCode` as the default editor/selector—across three professional benchmarks ([Section 3.1 and Appendix B](https://arxiv.org/abs/2610.14224)):

- **`WorkBuddy-Bench`** (test halves: $40$ `Code`, $35$ `Web`, $25$ `Office` tasks; scored by a composite verifier of deterministic checks plus `Gemini 3.5 Flash` LLM/VLM judge items),
- **`JobBench`** ($34$ test tasks across $7$ occupational domains; scored against a hidden weighted rubric by `Claude Opus 4.8`), and
- **`LAB` (Harvey Legal Agent Benchmark)** ($100$ held-out test tasks across $24$ legal practice areas and $4$ work types; scored for per-criterion pass rate `Partial` by `Claude Sonnet 4.6`, with no executable check).

All main-table numbers are means (with standard deviations in parentheses) over three independent runs with shuffled task orders ([Section 3.1, Tables 1–2](https://arxiv.org/abs/2610.14224)):

| Executor & Base Harness | Method | `WB-code` | `WB-web` | `WB-office` | `WB Avg.` | `JobBench` | `LAB Partial` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Gemini 3.5 Flash` + `OpenCode` | Base harness | $73.4\ (3.6)$ | $69.9\ (4.2)$ | $74.0\ (2.2)$ | $72.4\ (0.9)$ | $31.5\ (1.1)$ | $79.9\ (0.6)$ |
| `Gemini 3.5 Flash` + `OpenCode` | `ReasoningBank` | $72.6\ (3.1)$ | $70.7\ (6.9)$ | $76.8\ (2.1)$ | $73.4\ (2.5)$ | $30.8\ (2.3)$ | $77.3\ (3.1)$ |
| `Gemini 3.5 Flash` + `OpenCode` | `Meta-Harness` (offline) | $75.5\ (1.7)$ | $75.1\ (0.6)$ | $78.6\ (1.4)$ | $76.4\ (0.7)$ | $35.8\ (1.0)$ | $81.6\ (3.9)$ |
| `Gemini 3.5 Flash` + `OpenCode` | `Harness Scaling` | $74.0\ (1.1)$ | $74.0\ (1.9)$ | $77.9\ (1.3)$ | $75.3\ (0.7)$ | $32.9\ (3.1)$ | $78.9\ (4.3)$ |
| `Gemini 3.5 Flash` + `OpenCode` | **`Harness²` ($k=1$)** | **$76.5\ (1.3)$** | **$76.4\ (4.5)$** | **$79.5\ (3.1)$** | **$77.5\ (2.5)$** | **$36.0\ (2.3)$** | **$81.9\ (0.6)$** |
| `Gemini 3.5 Flash` + `OpenCode` | **`+ Recursion` ($k=5$)** | **$80.8\ (+4.3)$** | **$78.3\ (+1.9)$** | **$83.4\ (+3.9)$** | **$80.8\ (+3.3)$** | **$39.0\ (+3.0)$** | **$84.5\ (+2.6)$** |
| `Gemini 3.5 Flash` + `Codex` | Base harness | $74.0\ (1.7)$ | $72.5\ (1.2)$ | $77.0\ (2.2)$ | $74.5\ (1.6)$ | $33.9\ (2.6)$ | $71.6\ (1.3)$ |
| `Gemini 3.5 Flash` + `Codex` | `ReasoningBank` | $73.5\ (3.3)$ | $65.7\ (2.2)$ | $78.4\ (0.9)$ | $72.5\ (2.2)$ | $30.2\ (2.7)$ | $74.8\ (0.9)$ |
| `Gemini 3.5 Flash` + `Codex` | `Meta-Harness` (offline) | $74.6\ (3.3)$ | $67.8\ (1.5)$ | $77.6\ (1.6)$ | $73.3\ (0.8)$ | $34.8\ (1.6)$ | **$76.5\ (2.1)$** |
| `Gemini 3.5 Flash` + `Codex` | `Harness Scaling` | $75.5\ (2.1)$ | $69.3\ (3.8)$ | $76.7\ (3.8)$ | $73.8\ (1.7)$ | $33.6\ (0.6)$ | $67.9\ (2.4)$ |
| `Gemini 3.5 Flash` + `Codex` | **`Harness²` ($k=1$)** | **$79.4\ (3.6)$** | **$74.7\ (3.8)$** | **$78.5\ (2.4)$** | **$77.5\ (2.0)$** | **$35.3\ (3.5)$** | $75.6\ (4.5)$ |
| `Gemini 3.5 Flash` + `Codex` | **`+ Recursion` ($k=5$)** | **$81.0\ (+1.6)$** | **$77.3\ (+2.6)$** | **$82.8\ (+4.3)$** | **$80.4\ (+2.9)$** | **$39.5\ (+4.2)$** | **$78.5\ (+2.9)$** |
| `GLM-5` + `OpenCode` | Base harness | $58.4\ (5.2)$ | $59.2\ (6.9)$ | $75.1\ (3.2)$ | $64.2\ (1.4)$ | $20.6\ (2.0)$ | $59.6\ (2.7)$ |
| `GLM-5` + `OpenCode` | `ReasoningBank` | $63.3\ (4.3)$ | $61.4\ (4.2)$ | $72.0\ (0.9)$ | $65.6\ (1.2)$ | $20.7\ (1.2)$ | $71.3\ (1.3)$ |
| `GLM-5` + `OpenCode` | `Meta-Harness` (offline) | $69.0\ (1.2)$ | $61.9\ (4.1)$ | $78.1\ (1.3)$ | $69.7\ (0.6)$ | $20.9\ (2.3)$ | $60.0\ (1.1)$ |
| `GLM-5` + `OpenCode` | `Harness Scaling` | $60.3\ (2.6)$ | $63.4\ (2.2)$ | $79.8\ (3.0)$ | $67.8\ (1.3)$ | $19.3\ (2.2)$ | $67.8\ (2.8)$ |
| `GLM-5` + `OpenCode` | **`Harness²` ($k=1$)** | **$77.1\ (2.7)$** | **$69.1\ (3.3)$** | **$80.8\ (2.6)$** | **$75.7\ (2.2)$** | **$24.2\ (0.9)$** | **$71.4\ (2.9)$** |
| `GLM-5` + `OpenCode` | **`+ Recursion` ($k=5$)** | **$81.9\ (+4.8)$** | **$74.4\ (+5.3)$** | **$82.5\ (+1.7)$** | **$79.6\ (+3.9)$** | **$28.4\ (+4.2)$** | **$78.1\ (+6.7)$** |
| `GLM-5` + `Codex` | Base harness | $56.8\ (2.7)$ | $59.6\ (1.0)$ | $76.9\ (2.4)$ | $64.4\ (1.2)$ | $23.6\ (2.3)$ | $58.1\ (2.3)$ |
| `GLM-5` + `Codex` | `ReasoningBank` | $50.2\ (3.8)$ | $66.3\ (2.5)$ | $75.8\ (4.2)$ | $64.1\ (0.9)$ | $22.8\ (3.9)$ | $63.8\ (3.3)$ |
| `GLM-5` + `Codex` | `Meta-Harness` (offline) | **$76.8\ (1.9)$** | $67.8\ (4.1)$ | $77.9\ (2.8)$ | $74.2\ (1.6)$ | $23.9\ (1.4)$ | $59.0\ (1.4)$ |
| `GLM-5` + `Codex` | `Harness Scaling` | $49.5\ (3.9)$ | $60.9\ (1.7)$ | $78.2\ (2.4)$ | $62.9\ (1.3)$ | $17.5\ (3.3)$ | $63.7\ (2.8)$ |
| `GLM-5` + `Codex` | **`Harness²` ($k=1$)** | $74.9\ (1.5)$ | **$70.0\ (4.3)$** | **$81.1\ (2.0)$** | **$75.3\ (1.3)$** | **$24.8\ (0.2)$** | **$65.9\ (2.7)$** |
| `GLM-5` + `Codex` | **`+ Recursion` ($k=5$)** | **$77.2\ (+2.3)$** | **$73.3\ (+3.3)$** | **$84.3\ (+3.2)$** | **$78.3\ (+3.0)$** | **$27.9\ (+3.1)$** | **$70.7\ (+4.8)$** |

[Xu et al., 2026, Section 3.2, Tables 1 and 2; `+ Recursion` rows show the $k=5$ parallel recursion mean alongside its delta over $k=1$](https://arxiv.org/abs/2610.14224).

Three comparisons clarify *why* the contrastive mechanism matters:

- **Where the headline single-step gains occur:** At $k=1$, `Harness²` improves over the base harness in all $12$ benchmark–configuration pairs ([Section 3.2, Tables 1–2](https://arxiv.org/abs/2610.14224)). The abstract's "up to $11.5$ points on `WorkBuddy-Bench`" is `GLM-5` under `OpenCode` ($64.2 \to 75.7$ macro-average, driven by a $+18.7$-point jump on `WB-code` from $58.4$ to $77.1$, while average execution steps stay flat at $13.5 \to 13.7$). The "up to $4.5$ points on `JobBench`" is `Gemini 3.5 Flash` under `OpenCode` ($31.5 \to 36.0$), and the "up to $11.8$ percentage points on `LAB`" is `GLM-5` under `OpenCode` ($59.6\% \to 71.4\%$ `Partial`). With a single step, `GLM-5` ($75.7$ `OpenCode`, $75.3$ `Codex` on `WorkBuddy-Bench`) surpasses the unrefined `Gemini 3.5 Flash` base harnesses ($72.4$ and $74.5$).
- **Why unstructured test-time revision (`Harness Scaling`) often regresses:** `Harness Scaling` (Wang et al., 2026a) also spends test-time rollouts rewriting the same $8$-component tree without a verifier, but uses a single unscoped revision session per iteration with no composition gate and no cross-task library ([Appendix C](https://arxiv.org/abs/2610.14224)). Across Tables 1 and 2, `Harness Scaling` **underperforms the unmodified base harness in $7$ of the $12$ benchmark–configuration pairs** (for instance, dropping `GLM-5` + `Codex` on `WB-code` from $56.8$ to $49.5$ and on `JobBench` from $23.6$ to $17.5$), and regresses when scaled from $k=2$ to $k=3$ on both `WB-code` and `WB-office` ([Section 3.2 and Section 4, Figure 5](https://arxiv.org/abs/2610.14224)). Similarly, retrieval-only `ReasoningBank` falls up to $6.6$ points below the base harness on coding (`GLM-5` + `Codex` `WB-code`, $56.8 \to 50.2$).
- **Isolating the task-level ingredients (`Table 3`):** On `Gemini 3.5 Flash` (`OpenCode`, $k=1$), removing `INV`/`DOM` scoping drops `WorkBuddy-Bench` by $2.9$ points ($77.5 \to 74.6$) and `JobBench` by $3.7$ points ($36.0 \to 32.3$); skipping `Compose` and simply picking the better probe costs $1.6$ points ($77.5 \to 75.9$) and $1.9$ points ($36.0 \to 34.1$) ([Section 4, Table 3](https://arxiv.org/abs/2610.14224)). Nor do the gains require a stronger external editor: when the *same* model acts as both executor and editor at $k=2$ under `OpenCode` ([Section 4, Figure 8](https://arxiv.org/abs/2610.14224)), all five tested models improve on every benchmark, including `GLM-5` ($+13.7$ points on `LAB`) and `Claude Opus 5` ($+3.3$ on `WorkBuddy-Bench`, $+5.5$ on `JobBench`, and $+3.0$ on `LAB`).

## 3. What transfers: lessons vs. customized harnesses

### 3.1 Why each new task resets to the shared baseline harness

Suppose task $q_t$ finishes and yields an adopted harness $\mathcal{H}^\star_t$. A natural temptation is to set $\mathcal{H}_0^{(t+1)} \leftarrow \mathcal{H}^\star_t$ so the next task inherits the customized harness directly. `Harness²` explicitly rejects this design: **every task $q_{t+1}$ resets its execution harness to the shared baseline $\mathcal{H}_0$, while the cross-task state lives entirely inside the improvement harness $\mathcal{R}_{t+1}$** ([Section 2.1, Section 2.3, and Equation (4)](https://arxiv.org/abs/2610.14224)):

$$
\mathcal{R}_{t+1} = \operatorname{Update}(\mathcal{R}_t, \mathcal{H}^\star_t, \mathcal{D}_t).
$$

Why must the execution harness reset to $\mathcal{H}_0$ while the improvement harness accumulates $\mathcal{R}_{t+1}$? The stage prompts in Appendix H reveal the exact architectural reason: **what makes a harness strong on task $q_t$ is a mixture of reusable domain procedure and non-transferable workspace facts**.

During task $q_t$, Prompt 1 (`Harness Editor`) instructs the proposer that verified workspace facts discovered by the base rollout—such as *"in this workspace, attrition role `'HR'` in file A maps to position `'HR Manager'` in file B; `'Manager'` has no open requisition"*—are the highest-leverage edits to encode for *this* task's rerun, because they save the solver from rediscovering brittle file joins ([Appendix H, Prompt 1, p. 30–31](https://arxiv.org/abs/2610.14224)). On task $q_{t+1}$, however, those exact workspace facts are false anchors. As Prompt 1 warns the editor when reading earlier harnesses in `context/library/`: *"an earlier task's mapping or threshold is a hypothesis here, not a fact"* ([Appendix H, Prompt 1, p. 31](https://arxiv.org/abs/2610.14224)). Furthermore, every always-on rule carried over into `systemprompt.md`, `guardrails.md`, or `LongTermMEMORY.md` taxes the context window on every turn. Resetting the execution harness to $\mathcal{H}_0$ prevents task-$t$ workspace facts and bloated instructions from polluting task $q_{t+1}$.

### 3.2 How the Experience Reflector filters out confounded edits

Instead of inheriting $\mathcal{H}^\star_t$ as the runtime state, `Harness²` runs an **Experience Reflector** over the complete improvement record $\mathcal{D}_t$—which contains the base rollouts, *all* candidate harnesses (both adopted and rejected), their probe trajectories, the composition decisions, and the selected output ([Section 2.3 and Appendix H, Prompt 6](https://arxiv.org/abs/2610.14224)).

Prompt 6 enforces two strict rules on what enters `EXPERIENCE.md` inside $\mathcal{R}_{t+1}$:

1. **Selection does not imply attribution:** Because the terminal selector picked a deliverable with the harnesses hidden, *"the choice says which output read best — not which edit worked"* ([Appendix H, Prompt 6, p. 36](https://arxiv.org/abs/2610.14224)). An edit inside the winning harness $\mathcal{H}^\star_t$ may have contributed nothing, while an edit inside a rejected candidate may have fixed a concrete failure mode.
2. **Explicit `confounded` verdicts and `NO_ENTRY` abstention:** The reflector diffs every candidate against `context/base_harness/` and assigns each modified component one of four verdicts: `helped`, `hurt`, `no observed effect`, or `confounded`. It is instructed to prefer `confounded` whenever a candidate modified multiple components simultaneously and no probe isolates one of them, or when a single rollout cannot separate the edit from sampling noise. A transferable lesson (`cue`, `lesson` naming the specific component, `evidence`, `confidence`) may be written **only** if supported by non-confounded execution contrasts; otherwise the reflector outputs `NO_ENTRY` ([Section 2.3 and Appendix H, Prompt 6, p. 36–37](https://arxiv.org/abs/2610.14224)).

### 3.3 Empirical evidence for lesson transfer vs. file copying

The paper's provenance, ablation, and transfer experiments show both what the library contributes and how differently it behaves from direct harness inheritance:

- **Adaptation dominates verbatim copying (`Figure 10`):** When Xu et al. (2026) measure line-level containment between modified component files in adopted harnesses and the library snapshot available at refinement time (excluding boilerplate lines shared by $>20\%$ of a benchmark's harnesses), library provenance appears in **$22\%$** of modified files on `LAB` ($6\%$ reused $\ge 50\%$ of lines, $16\%$ adapted $5\text{–}50\%$) and `WB-office` ($6\%$ reused, $16\%$ adapted), **$37\%$** on `WB-code` ($7\%$ reused, $30\%$ adapted), **$46\%$** on `WB-web` ($10\%$ reused, $36\%$ adapted), and **$47\%$** on `JobBench` ($18\%$ reused, $29\%$ adapted, with $53\%$ fresh $<5\%$; note that the stacked bar in Figure 10 left prints `18` and `43` alongside `53` fresh, summing to $114$, whereas the text in Appendix E.2 states library provenance is $47\%$ on `JobBench`, matching $100\% - 53\% = 47\%$). Across all benchmarks, **only $2\text{–}10\%$** (up to $18\%$ on `JobBench` in the figure) of modified files copy at least half their lines from an earlier harness; most edits are written fresh ($53\text{–}78\%$) or substantially adapted ($16\text{–}36\%$), with `C6` skills and `C7` scripts accounting for roughly half of all edits ([Appendix E.2, Figure 10](https://arxiv.org/abs/2610.14224)).
- **Online library ablation (`Table 3`):** On `Gemini 3.5 Flash` (`OpenCode`, $k=1$), disabling the improvement library (`w/o library`, so each task still runs `Propose–Probe–Compose` at $k=1$ but neither reads from nor writes to $\mathcal{R}_t$) lowers `WorkBuddy-Bench` from $77.5$ to $74.4$ ($-3.1$ points) and `JobBench` from $36.0$ to $34.2$ ($-1.8$ points) ([Section 4, Table 3](https://arxiv.org/abs/2610.14224)). Even without the library, $k=1$ contrastive probing beats the base harness ($74.4$ vs. $72.4$ on `WorkBuddy-Bench`; $34.2$ vs. $31.5$ on `JobBench`), and adding the library contributes a further $+3.1$ and $+1.8$ points.
- **Unprobed direct editing with accumulated libraries (`Figure 7 left`):** While direct one-shot editing of $\mathcal{H}_0$ *without* probes or a library collapses on `WB-code` ($73.4 \to 63.2$) and `WB-web` ($69.9 \to 65.0$), conditioning that same zero-probe edit on a library accumulated at $k=1$ restores base performance ($73.8$ on `WB-code`, $72.6$ on `WB-web`, $77.5$ on `WB-office`), and conditioning on a library accumulated under $k=5$ parallel exploration yields net gains over the base harness without running a single probe on the test task: **$74.3$ ($+0.9$)** on `WB-code`, **$74.3$ ($+4.4$)** on `WB-web`, and **$79.1$ ($+5.1$)** on `WB-office` ([Section 4, Figure 7 left](https://arxiv.org/abs/2610.14224)). Deeper task-level exploration on earlier tasks thus produces higher-quality lessons for future zero-probe edits.
- **Cross-benchmark and leave-one-out transfer (`Figure 7 right` and `Figure 11`):** Initializing `WorkBuddy-Bench` ($k=1$) with an improvement library transferred from `JobBench` adds $+0.6$ points on `WB-code` ($76.5 \to 77.1$), $+0.5$ on `WB-web` ($76.4 \to 76.9$), and $+2.9$ on `WB-office` ($79.5 \to 82.4$) over standard $k=1$ `Harness²` ([Section 4, Figure 7 right](https://arxiv.org/abs/2610.14224)). In an offline leave-one-out (`LOO`) test where each task is re-run at $k=1$ using the completed stream's library with its own entry removed, scores rise from $77.5 \to 79.4$ ($+1.9$) on `WorkBuddy-Bench` and from $36.0 \to 37.7$ ($+1.7$) on `JobBench`—though the authors explicitly note that `LOO` accesses later-task experience and repeated execution, making it an offline reuse check rather than an online deployment gain ([Section 4 and Appendix E.2, Figure 11](https://arxiv.org/abs/2610.14224)).

## 4. Caveats: comparable reruns, model-judged evaluation, and shared biases

The empirical gains of `Harness²` come with structural assumptions and evaluation limits that the paper documents directly in Sections 2.2.1, 3.1, Appendix B, Appendix E.6, and Appendix F. Both caveats below are central to interpreting what the results do and do not prove.

### 4.1 The method assumes tasks can be rerun in isolated, comparable environments

Contrastive probing only produces valid evidence if executing $\mathcal{H}_{\mathrm{INV}}$ and $\mathcal{H}_{\mathrm{DOM}}$ on task $q_t$ tests those harnesses under **the same initial state and external conditions** as the reference rollout $e_0$ ([Section 2.2.1 and Appendix A.1](https://arxiv.org/abs/2610.14224)).

In `Harness²`'s benchmark suite, this assumption holds by construction: every rollout mounts the harness into a **fresh container** over static workspace files ([Appendix B, p. 18–19](https://arxiv.org/abs/2610.14224)). At budget $k=1$, one task requires $4$ full agent executions ($1$ base reference rollout, $2$ candidate probes, and $1$ composed rollout) plus editor, composer, and reflector sessions; at $k=5$ in parallel recursion, one task executes $k = 5$ base rollouts, $2k = 10$ candidate probes, and $k = 5$ composed rollouts ($20$ total agent executions per task), spending roughly $10^1\text{–}10^2$ million total tokens per task on `JobBench` and $10^0\text{–}10^1$ million tokens on `WorkBuddy-Bench` ([Section 2.2.2, Section 4, Figure 6, and Appendix F](https://arxiv.org/abs/2610.14224)).

In real open-world deployments, two obstacles limit this rerun protocol:

- **Non-idempotent or stateful actions:** If a task mutates external production state—sending emails, executing financial transactions, modifying live databases, or interacting with a human user whose responses shift across turns—the task cannot be rewound into $4$ to $20$ identical containers without a high-fidelity sandbox or mock environment.
- **Stochastic outcome variance across single probes:** Even inside fresh containers, each candidate harness in parallel recursion is probed **only once** ([Appendix H, Prompt 3](https://arxiv.org/abs/2610.14224)). As the failure analysis in Appendix E.6 (`Figure 14`, *"Neither the edit nor the pick"*) shows on `JobBench` (`web administrators`), when a criterion is bimodal across stock rollouts (scoring $36, 36, 36, 24, 24$ across five unmodified runs), a single probe draw can misattribute sampling noise to a harness edit ($31 \to 24$, $-7$ points).

### 4.2 Model-judged evaluation, overlapping editor/judge families, and run-to-run variance

Second, almost all benchmark scoring in the paper relies on **LLM or VLM judges**, and the models acting as harness editor/selector overlap closely with the models acting as benchmark judges ([Section 3.1 and Appendix B](https://arxiv.org/abs/2610.14224)):

| Role in Pipeline / Benchmark | Model Used in Paper | Paper Location |
| --- | --- | --- |
| Default **Harness Editor**, **Composer**, **Blind Selector**, and **Reflector** | `Claude Opus 4.8` (under `OpenCode`, high reasoning effort) | Section 3.1, p. 7 & Appendix D.2, p. 21 |
| **`JobBench`** evaluator (weighted rubric judge; no executable check) | `Claude Opus 4.8` | Appendix B, p. 19 |
| **`LAB`** evaluator (per-criterion pass rate judge; no executable check) | `Claude Sonnet 4.6` | Appendix B, p. 19 |
| **`WorkBuddy-Bench`** evaluator (composite: deterministic tests + LLM/VLM judge) | Deterministic checks + `Gemini 3.5 Flash` judge | Appendix B, p. 18–19 |
| **Executors** evaluated in main tables | `Gemini 3.5 Flash` and `GLM-5` (plus self-editing in Fig. 8) | Section 3.1, p. 7 & Section 4, Fig. 8 |

Although a workspace guard prevents the editor and selector from ever seeing the grading rubric or judge scores ([Appendix B and Appendix D.2](https://arxiv.org/abs/2610.14224)), **the editor/selector (`Claude Opus 4.8`) and the `JobBench` judge (`Claude Opus 4.8`) are the exact same model** (and on `LAB`, the judge `Claude Sonnet 4.6` is from the same model family; on `WorkBuddy-Bench`, the judge `Gemini 3.5 Flash` matches one of the two main executors). Shared stylistic priors, structural preferences, or blind spots between the editor/selector and the evaluator can therefore inflate perceived gains on rubric-judged deliverables.

Moreover, Appendix E.6 (`Figure 14`) and the variance parentheses in Tables 1–2 show why **small differences—especially modest cross-benchmark transfer gains—must be treated with statistical caution**:

- **Run-to-run standard deviations often exceed $1\text{–}4$ points:** Main tables report means and SDs over only three shuffled-stream runs ([Section 3.1](https://arxiv.org/abs/2610.14224)). On `WorkBuddy-Bench`, domain-level SDs for `Harness²` ($k=1$) range from $1.3$ to $4.5$ points (with macro-average SDs of $1.3$ to $2.5$ points; [Table 1](https://arxiv.org/abs/2610.14224)). On `JobBench` and `LAB`, SDs reach up to $3.5$ and $4.5$ points (`Gemini 3.5 Flash` + `Codex`, $35.3\ (3.5)$ and $75.6\ (4.5)$; [Table 2](https://arxiv.org/abs/2610.14224)). Against run-to-run SDs of $1.3\text{–}4.5$ points (and with no error bars reported on the bar charts in Figures 1, 5, 7, 8, 11, 12, and 13), small transfer increments—such as the $+0.6$ (`WB-code`) and $+0.5$ (`WB-web`) gains from transferring the `JobBench` library in Figure 7 (right), or the $+0.9$ (`WB-code`) zero-probe library gain in Figure 7 (left)—lie well within one standard deviation of single-stream noise.
- **Shared model biases can lock in restrictive rules or selector errors (`Figure 14`):** Because the proposer, composer, and selector share model priors without ground-truth checks, an incorrect premise from the base trajectory can be codified into a skill that caps every candidate in the pool—for example, forbidding header lowercasing on `WB-code` (`security · header injection`, $50 \to 42$, $-8$ points), writing a self-confirming checker script around an `id()`-keyed cache assumption (`validator cache`, $75 \to 64$, $-11$ points), or pre-deciding a tax adjustment rule on `LAB` (`notice of deficiency`, $83 \to 81$, $-2$ points). Likewise, the blind selector can prefer a cleaner-looking deliverable over one that satisfies hidden static checks or broad rubric coverage (`WB-code` `blog_tags`: picks a $55$-scoring output over an already-executed $95$, dropping $96 \to 55$; `LAB` `transfer pricing`: picks precision over coverage, dropping $54 \to 26$). Even judge sessions themselves can fail: on `WB-web` (`particle boundary`), three judge sessions timed out without returning a verdict, which the benchmark harness counted as failures ($44 \to 10$, $-34$ points; [Appendix E.6, Figure 14](https://arxiv.org/abs/2610.14224)).

## 5. Takeaways for practitioners evolving agent harnesses

If you are building an agent harness that adapts at test time across open-ended tasks, `Harness²` ([Xu et al., 2026](https://arxiv.org/abs/2610.14224)) offers five concrete engineering lessons:

1. **Do not rewrite a live harness without a matched behavioral probe.** Unprobed, memory-free edits to a base harness degrade coding and web performance steeply (`WB-code` $73.4 \to 63.2$, `WB-web` $69.9 \to 65.0$; Figure 7 left), and unscoped single-chain revision (`Harness Scaling`) loses to the unmodified base harness in $7$ of $12$ benchmark configurations (Tables 1–2). Treat `KEEP_BASE` as the default unless a matched rerun demonstrates a concrete trajectory fix.
2. **Split failure diagnosis into orthogonal `INV` (workflow/tool) and `DOM` (domain contract) scopes, then compose selectively.** Separating *how the agent executes* (tool errors, verification scripts, output watchdogs) from *what a domain practitioner checks* (grain of aggregation, source hierarchy, redline survivability) prevents a single editor from conflating process bugs with domain rules ($+2.9$ to $+3.7$ points in Table 3).
3. **Prefer on-demand skills, deterministic scripts, and tool-loop hooks over always-on prompt prose.** Always-on files (`systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`) tax every model call. In `Harness²`, roughly half of all edits land in on-demand `C6` skills and `C7` scripts (Figure 10 right), and removing skills or scripts from the evolved harness costs $5.3$ and $3.7$ average points on `WorkBuddy-Bench` versus $1.5$ points for removing prompt-group edits (Appendix E.1, Table 6). When an agent repeatedly ignores prose instructions (such as researching until the turn budget expires), interpose a lightweight `C8` middleware hook in the tool loop rather than adding louder prompt warnings (Appendix E.7, Figure 19).
4. **Reset the execution harness to $\mathcal{H}_0$ on each new task and transfer contrast-backed lessons instead.** Never let task-$t$ workspace facts silently become task-$(t+1)$ defaults. Store past harnesses as inspectable examples alongside an attribution log (`helped` / `hurt` / `confounded`), require the editor to re-verify any borrowed mapping against the new workspace's trajectories, and abstain (`NO_ENTRY`) when multi-component edits are confounded.
5. **Separate how a harness evolves from what the harness is permitted to change.** `Harness²` already has to guard one boundary in prompt prose—warning editors that *"the deliverable's shape belongs to the request, not to you"* ([Appendix H, Prompt 1, p. 32](https://arxiv.org/abs/2610.14224)) and using a runtime guard so improvement sessions cannot access evaluator state ([Appendix D.2](https://arxiv.org/abs/2610.14224)). More broadly, evolving an execution harness and deciding what responsibilities or task contracts must stay *outside* the editable harness are complementary questions—a perspective explored from the governance and contract side in [ProContract](/projects/procontract/).
