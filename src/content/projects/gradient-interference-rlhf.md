---
title: "Gradient Interference in RLHF"
tagline: "Minority collapse in single-reward RLHF can appear while the group rewards still agree."
description: "A controlled study of group-aware RLHF. It separates value-level preference conflict from optimization-level gradient interference, and compares worst-group objectives with explicit gradient coordination."
status: "Manuscript"
period: "2026"
citationYear: 2026
order: 1
tags:
  - RLHF
  - Group Alignment
  - Gradient Interference
  - Reinforcement Learning
pending:
  - arXiv coming soon
  - Code coming soon
bibtex: |
  @misc{cui2026gradientinterference,
    author    = {Cui, Mingxuan and Zhang, Yuji and Zhou, Duo},
    title     = {Not All Reward Heterogeneity is Value Conflict: Gradient Interference in Group-Aware {RLHF}},
    year      = {2026},
    url       = {https://www.duo-zhou.com/projects/gradient-interference-rlhf/}
  }
---

**Not All Reward Heterogeneity is Value Conflict: Gradient Interference in Group-Aware RLHF**

Mingxuan Cui, Yuji Zhang, **Duo Zhou**

arXiv coming soon · Code coming soon

## Two ways a minority group can be left behind

Standard RLHF fits one reward to pooled comparisons, then optimizes a policy against that reward. When preferences differ, the pooled reward can serve one group well and leave another poorly aligned. The question is which mechanism produces that disparity.

This paper separates two of them:

| Failure | What has gone wrong |
| --- | --- |
| Value-level conflict | Raising one group's reward requires lowering the other's. |
| Optimization-level interference | The policy update points against a group's reward-improving direction. |

The two can come apart. Group rewards can still move together while the update computed from the pooled objective works against the smaller group.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig1-worst-group.png" alt="Worst-group reward against preference divergence for vanilla RLHF, MaxMin, SumGrad, and PCGrad. Vanilla drops sharply while rewards are still positively correlated. PCGrad stays high longer.">
  <figcaption>Worst-group reward as the two group preferences are pulled apart. Vanilla RLHF falls off while the rewards are still positively correlated. PCGrad holds up further into the conflicting region. Shaded bands mark where the group rewards agree, are orthogonal, or conflict.</figcaption>
</figure>

## Turn agreement into conflict with one parameter

The benchmark is a controlled IMDb review task with a fixed 80/20 population split. The majority reward emphasizes positive sentiment. The minority reward emphasizes conciseness. A mixing weight $\alpha \in [0.5, 1.3]$ sets how far apart those rewards are. At $\alpha = 0.5$ the two rewards are the same. At $\alpha = 1$ each reward is one raw attribute. For $\alpha > 1$ improving one attribute explicitly penalizes the other.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig4-pearson.png" alt="Pearson correlation between the two group rewards, falling from 1 at alpha 0.5, through about 0.65 near alpha 0.75, to negative values after alpha 1.">
  <figcaption>Correlation between the majority and minority rewards, computed on a fixed pool of sentiment and conciseness scores. The curve is still positive near $\alpha = 0.75$, where vanilla RLHF collapses, and crosses zero only at $\alpha = 1$.</figcaption>
</figure>

Single-reward RLHF keeps a high majority reward across this sweep. The minority reward drops sharply as $\alpha$ moves from $0.7$ to $0.75$. At that collapse the group rewards are still positively correlated across the completion pool, with Pearson $r = 0.65$, because short, positive reviews satisfy both groups. The failure is that global reward correlation on a static pool does not imply local gradient alignment around the current policy. Under an 80/20 split, the steepest direction for the pooled reward is to pad generations with extra positive phrasing, which lengthens the output and points directly against the minority conciseness gradient. The sampled text becomes long, repetitive, and strongly positive as the pooled objective locks onto the majority attribute.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig2-vanilla-groups.png" alt="Vanilla RLHF minority and majority reward versus preference divergence. Minority reward collapses between alpha 0.7 and 0.8 while majority reward stays high and then rises.">
  <figcaption>Vanilla RLHF on the same sweep, with the minority and majority rewards drawn separately. The minority curve breaks between $\alpha = 0.7$ and $0.8$. The majority curve does not.</figcaption>
</figure>

Group-aware methods, trained under the same preference settings, avoid that collapse and keep a comparable majority reward. The disparity is in how the aggregated reward becomes an update.

## The harmful step happens early

Near the collapse, cosine similarity between the KL-regularized PPO update and the minority reward gradient falls below $-0.5$ early in training, at $\alpha = 0.75$ and $\alpha = 0.8$.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig3-cosine-kl.png" alt="Cosine similarity between the KL-regularized PPO update and the minority gradient over 800 PPO steps, for alpha 0.7, 0.75, and 0.8. The 0.75 and 0.8 curves go negative early, then return toward positive.">
  <figcaption>Cosine similarity between the KL-regularized PPO update and the minority reward gradient. At $\alpha = 0.75$ and $\alpha = 0.8$ the update points against the minority gradient early, then the similarity comes back toward positive while the minority reward stays low.</figcaption>
</figure>

The same early anti-alignment remains when the KL term is left out of the cosine, so the regularizer does not account for the whole effect. Later in training the similarity turns positive again, yet the minority reward stays low. Once early conflicting steps push the generator into long, repetitive completions, on-policy rollout loses support on concise text; local gradients that stop opposing conciseness inside that collapsed region cannot recover a mode the policy no longer samples.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig5-cosine-no-kl.png" alt="Cosine similarity between the vanilla PPO gradient without the KL term and the minority gradient. The early negative dip at alpha 0.75 and 0.8 remains.">
  <figcaption>The same comparison with the KL term removed from the vanilla gradient. The early negative dip at $\alpha = 0.75$ and $\alpha = 0.8$ is still there.</figcaption>
</figure>

## Using every group is different from coordinating their gradients

The comparisons hold the preference data fixed and change only the update rule:

| Method | Update rule | What changes |
| --- | --- | --- |
| Vanilla RLHF | One reward fit to the pooled data | The single-reward baseline. |
| MaxMin-RLHF | Optimize the current worst group | Which objective is active. The active group switches as relative scores cross. |
| SumGrad | Add the two group PPO gradients | Both learning signals are kept, and their directions are left unchanged. |
| PCGrad-PPO | Remove conflicting components, then combine those same gradients | The part of each update that opposes the other group is taken out. |
| MGDA-PPO | Minimum-norm convex combination of the group gradients | A second coordination rule, so the result is not tied to one projection formula. |

MaxMin-RLHF raises worst-group reward once vanilla RLHF begins to fail, including while the rewards are still positively correlated. It does that by training against whichever group is currently behind. SumGrad holds through moderate divergence. Once the preferences conflict more strongly, adding the gradients is not enough.

<div class="figure-row">
  <figure>
    <img src="/figures/gradient-interference-rlhf/fig9-minority.png" alt="Minority reward versus preference divergence. Vanilla collapses after alpha 0.7. SumGrad and MaxMin hold until about alpha 1.1, then fall. PCGrad stays high until alpha 1.2.">
    <figcaption>Minority reward. Summing the two group gradients holds the minority up through moderate divergence. Past $\alpha = 1.1$, that sum falls with MaxMin, and PCGrad is the curve that stays up longer.</figcaption>
  </figure>
  <figure>
    <img src="/figures/gradient-interference-rlhf/fig10-majority.png" alt="Majority reward versus preference divergence. Vanilla stays high. MaxMin dips near alpha 1.0 to 1.1. PCGrad stays flatter.">
    <figcaption>Majority reward on the same runs. Vanilla keeps this score high, including through the minority collapse. MaxMin gives some of it up near $\alpha = 1.0$. PCGrad moves less.</figcaption>
  </figure>
</div>

Worst-group reward on the sweep:

| $\alpha$ | MaxMin | SumGrad | PCGrad | MGDA |
| --- | ---: | ---: | ---: | ---: |
| 0.6 | 0.926 | 0.939 | 0.949 | 0.937 |
| 0.8 | 0.841 | 0.897 | 0.910 | 0.924 |
| 1.0 | 0.693 | 0.822 | 0.903 | 0.912 |
| 1.1 | 0.729 | 0.819 | 0.869 | 0.825 |
| 1.2 | 0.181 | −0.106 | 0.835 | 0.842 |
| 1.3 | −0.272 | −0.073 | 0.432 | 0.536 |

At $\alpha = 1.2$, SumGrad's worst-group reward is $-0.106$. PCGrad-PPO reaches $0.835$ and MGDA-PPO reaches $0.842$ from the same group gradients. MGDA is a second way of combining those gradients, a minimum-norm convex combination, so the recovery is not tied to PCGrad's projection. The line plots above do not include MGDA.

That minority collapse is easy to miss on an aggregate dashboard. Weighted by the 80/20 population split, overall utility dips briefly between $\alpha = 0.7$ and $0.9$ and then climbs back up as the policy over-optimizes the majority sentiment reward, masking the minority collapse behind a rising population mean.

<figure>
  <img src="/figures/gradient-interference-rlhf/fig8-overall.png" alt="Population utility, 80 percent majority reward plus 20 percent minority reward, versus preference divergence. Vanilla dips between alpha 0.7 and 0.9, then climbs back as the majority reward rises.">
  <figcaption>Population score, weighted 80% majority and 20% minority. Vanilla dips where the minority collapses, then climbs again because the majority reward keeps rising. The aggregate can look repaired while the minority score is still down.</figcaption>
</figure>

The training trajectories of MaxMin-RLHF at $\alpha = 1.2$ and $\alpha = 1.3$ show why scalar objective switching still breaks down under strong divergence. By always training on whichever group is currently behind, MaxMin successfully pulls the lagging group toward the leading group so that the two batch scores track each other. However, because each step still applies a single group's unprojected gradient, strongly opposing updates undo each other's progress and lock both groups at a low plateau ($0.181$ at $\alpha = 1.2$ and $-0.272$ at $\alpha = 1.3$), whereas projecting out the conflicting gradient component in PCGrad or MGDA keeps worst-group reward above $0.83$ at $\alpha = 1.2$.

<div class="figure-row">
  <figure>
    <img src="/figures/gradient-interference-rlhf/fig6-maxmin-12.png" alt="MaxMin training at alpha 1.2. Majority batch score starts near 0.4 and rises to meet the minority score near 1.1.">
    <figcaption>MaxMin training at $\alpha = 1.2$. The majority batch score starts behind and is pulled up toward the minority score.</figcaption>
  </figure>
  <figure>
    <img src="/figures/gradient-interference-rlhf/fig7-maxmin-13.png" alt="MaxMin training at alpha 1.3. Minority score starts higher, dips, and then the two group scores track each other.">
    <figcaption>MaxMin training at $\alpha = 1.3$. The minority score starts ahead, comes down, and the two groups then move together.</figcaption>
  </figure>
</div>

The collapse and the recovery both recur when the backbone changes. At $\alpha = 1.0$, with the same architecture used for the policy and the reward model, vanilla minority reward is $0.076$ on Pythia-160M and $0.027$ on Llama-3.2-1B. PCGrad-PPO reaches $0.968$ and $0.956$, while majority reward stays near the vanilla level.

| Backbone | Method | Minority | Majority |
| --- | --- | ---: | ---: |
| Pythia-160M | Vanilla | 0.076 | 0.653 |
| Pythia-160M | MaxMin | 0.957 | 0.619 |
| Pythia-160M | PCGrad | 0.968 | 0.654 |
| Llama-3.2-1B | Vanilla | 0.027 | 0.684 |
| Llama-3.2-1B | MaxMin | 0.902 | 0.695 |
| Llama-3.2-1B | PCGrad | 0.956 | 0.705 |

On one evaluation prompt at $\alpha = 1.2$, vanilla RLHF writes a 301-token repetitive review, with minority reward $-0.410$ and majority reward $1.228$. PCGrad-PPO answers the same prompt in 15 tokens, with minority reward $0.932$ and majority reward $1.000$. That pair is one illustration from the paper. The plots and tables above are the aggregate comparison.

## Scope

The groups are synthetic attributes on IMDb reviews, sentiment against conciseness, at an 80/20 split. They are task attributes. The main experiments use GPT-2 and two groups. The result is a diagnostic of one optimization failure in aggregated RLHF: a minority group can be left behind by the update rule while the group rewards are still compatible. Larger models, preference data collected from people, and settings with more than two groups remain open. Together with [Cumulative Fisher Protection](/projects/cumulative-fisher/) and [Geometric Unlearning](/projects/geometric-unlearning/), this work treats post-training tradeoffs as directional conflicts in parameter space rather than scalar weights in the loss.
