---
title: "Cumulative Fisher Protection"
tagline: "Alignment can move a long way if it stays off the directions a capability is sensitive to."
description: "An optimizer correction for RLHF and DPO. It estimates the initial policy's output Fisher on a capability and shrinks cumulative displacement along those directions, leaving the rest of the update free for alignment."
status: "Under review"
period: "2026"
citationYear: 2027
order: 2
tags:
  - RLHF
  - Capability Preservation
  - Fisher Geometry
  - Alignment
pending:
  - arXiv coming soon
  - Code coming soon
bibtex: |
  @misc{zhou2027cfp,
    author    = {Zhou, Duo},
    title     = {Where the Policy Moves Matters: Capability Preservation via Cumulative Fisher Protection in {RLHF}},
    year      = {2027},
    note      = {Under review},
    url       = {https://www.duo-zhou.com/projects/cumulative-fisher/}
  }
---

**Where the Policy Moves Matters: Capability Preservation via Cumulative Fisher Protection in RLHF**

Under review. arXiv coming soon · Code coming soon

## The penalty should have the shape of the capability

A KL penalty pulls the policy toward the reference on the prompts used for alignment. A capability lives on a different set of prompts, and it is sensitive to a different set of parameter directions. Shortening every step by the same amount, which is what a larger KL coefficient does, still lets the path wander along the directions that change math, code, or refusal behavior. InstructGPT already saw the practical version of this: raising the KL coefficient by two orders of magnitude left regressions on DROP and SQuAD in place and cut the reward.

Cumulative Fisher Protection (CFP) builds the penalty from the capability. At the initial policy, sample prompts from the capability and responses from the model, and form the output Fisher, the second moment of $\nabla_\theta \log \pi_{\theta_0}(y \mid x)$. Directions with large Fisher eigenvalues are the ones in which a small parameter step changes the output distribution. The correction is then a proximal map on the whole path from initialization, not a penalty on the latest step. In the diagonal form used in the experiments, coordinate $i$ with Fisher weight $f_{d,i}$ moves by

$$\theta_{t+1,i} = \theta_{0,i} + \frac{\theta_{t,i} - \theta_{0,i} + d_{t,i}}{1 + \lambda_t f_{d,i}}.$$

A sensitive coordinate is shrunk. A coordinate the capability barely uses keeps the optimizer's proposal. The base optimizer can be PPO or DPO. The Fisher is estimated once, from 128 calibration sequences, and then held fixed.

<figure>
  <img src="/figures/cumulative-fisher/cfp-fig1-geometry.png" alt="A loss surface for a protected capability. Standard RLHF climbs into a high-loss region. KL shortens that path. A Fisher-shaped update bends along the low-sensitivity direction and stays near the initial capability. A top view shows a round KL ball against an elongated Fisher ellipse.">
  <figcaption>Equal-length steps are not equal. The round set is a KL ball around the initial policy. The elongated set is a level curve of the capability Fisher. The protected update spends its budget in the flat direction.</figcaption>
</figure>

## What has to be measured is the path, not the step

Successive updates can reinforce or cancel. A single gradient that looks safe can still land, after a hundred steps, deep in a sensitive direction. The diagnostic is therefore the displacement $\Delta_t = \theta_t - \theta_0$ from the initial checkpoint. Sensitive drift is the Fisher-weighted length $\sqrt{\Delta_t^\top \hat F_d \Delta_t}$. Free drift is the ordinary length of $\Delta_t$ on the 99% of coordinates with the smallest Fisher weights. The intended picture is a curve that keeps moving right and stops climbing.

<figure>
  <img src="/figures/cumulative-fisher/cfp-fig2-drift.png" alt="Three plots of sensitive drift against free drift for math-anchored PPO, math-anchored DPO, and safety-anchored PPO. Fisher-anchored curves stay below the unregularized and KL baselines at similar free drift.">
  <figcaption>Cumulative displacement on Qwen2.5-7B-Instruct. Vertical is Fisher-weighted drift under the training anchor. Horizontal is movement on the low-weight coordinates. Points are initialization, step 64, and the final checkpoint. The anchored runs stay lower at a comparable free drift. KL barely separates from the unregularized path.</figcaption>
</figure>

At step 64 the quadratic version of that drift, relative to the unregularized run of the same optimizer, splits the methods cleanly. Each anchor suppresses its own domain the most. It also suppresses the others. A math anchor, built with no code prompts, still cuts code-sensitive drift by 40%.

| Training arm | Math Fisher | Safety Fisher | Code Fisher |
| --- | ---: | ---: | ---: |
| PPO + KL | −10% | +1% | −7% |
| PPO + math anchor | −67% | −32% | −40% |
| PPO + safety anchor | −46% | −53% | −34% |
| PPO + code anchor | −33% | −14% | −41% |
| DPO + math anchor | −95% | −93% | −63% |

KL is the control that shows why the shape matters. It changes the math drift by 10% and the safety drift by about zero. The anchor changes the math drift by 67%, with $\lambda = 1$ for PPO and $\lambda = 10$ for DPO.

## The scoreboard moves by a point. The path moves by a lot.

On Qwen2.5-7B-Instruct, unregularized PPO takes OlympiadBench from the initial 39.56% to 37.33%. The three anchors land between 37.93% and 38.96%. The math anchor reaches 38.5% and moves MATH from 75.30% to 75.48%. That is a partial recovery, and it should be read as one: the initial Olympiad score is 39.56%, and PPO with a KL penalty reaches 40.1% on that one benchmark while barely bending the sensitive path. A single cell can rise for a reason that has little to do with staying off the capability's sensitive directions.

Under DPO the geometry and the score move together. Math anchoring brings MATH from 74.88% back to the initial 75.36%, and OlympiadBench from 38.96% to 40.15%, above the initial 39.56%.

<figure>
  <img src="/figures/cumulative-fisher/cfp-fig3-profiles.png" alt="Three radar charts for math-anchored PPO, math-anchored DPO, and safety-anchored PPO. Each compares the initial model, the unregularized baseline, and the Fisher-anchored run across MATH, GSM8K, OlympiadBench, MMLU, LiveCodeBench, and safety.">
  <figcaption>Capability profiles. The dashed polygon is the initial model. Each axis has its own scale, so the area of a polygon is not a score. Safety is the reduction in harmful compliance. The math anchor pulls Olympiad and GSM8K outward. The safety anchor pulls the safety axis outward and leaves the others close to where they started.</figcaption>
</figure>

Safety is learned by PPO, and the anchors keep it. Harmful compliance on SORRY-Bench falls from 34.55% for the initial model to 13.18% for PPO. The math, code, and safety anchors finish at 14.77%, 16.59%, and 12.27%. The safety anchor is the lowest final checkpoint in the comparison, and it also reaches 25.12% on LiveCodeBench, just above the initial 24.74%. Relative to the initial policy, those three anchors retain about 93%, 84%, and 104% of PPO's reduction in harmful compliance. DPO does not buy that safety reduction: its harmful-compliance rate stays near the initial model, including with the math anchor. The capability result and the safety result belong to different optimizers.

<figure>
  <img src="/figures/cumulative-fisher/cfp-fig4-heatmap.png" alt="Heatmap of final MATH, GSM8K, OlympiadBench, MMLU, LiveCodeBench, and SORRY-Bench scores for the base model, PPO with and without KL and three Fisher anchors, and DPO with and without a math anchor. Color is the within-row rank.">
  <figcaption>Final checkpoints. Color is the rank within a row, not a significance mark. SORRY-Bench is harmful compliance, and lower is better on that row. GSM8K is where the PPO anchors move the most, from 81.3% initially and 82.0% after PPO to 84.5–85.3%. MATH and MMLU barely move.</figcaption>
</figure>

## What this geometry leaves out

The experiments use a diagonal Fisher, so interactions between coordinates are invisible, and a sensitive direction that lives in those interactions is missed. The 128 calibration prompts overlap the corresponding evaluation sets, so these numbers do not establish the same protection on unseen prompts. The correction can also be the wrong one when the initial policy is weak and the alignment update is trying to repair that same skill: shrinking the sensitive directions then protects the error. The claim the runs support is narrower. On this model, controlling cumulative displacement in the initial policy's output-sensitive coordinates preserves a selected capability while PPO still learns most of its safety reduction.
