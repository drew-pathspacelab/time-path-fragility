---
layout: single
title: "Beyond the Realized Path"
excerpt: "Most analysis focuses on the single path that occurred. Pathspace asks what changes when we explore the paths that could have occurred."
permalink: /writing/beyond-the-realized-path/
author: drew_profile
author_profile: true
---


## 1. Models assume a single history

Most of the systems we care about—financial markets, ecosystems, biological networks, even societies—are complex, adaptive, and fundamentally path dependent. Their present state depends not only on where they are, but on the sequence of events that brought them there.

Despite this, we usually study them through a single realized history. We collect observations, fit models, estimate relationships, and evaluate performance on the one sequence of events that actually occurred.

I&rsquo;ve done this myself many times. After all the statistical tests, regression diagnostics, and carefully constructed figures, it&rsquo;s easy to feel that the underlying dynamics have been uncovered. The analyses are technically rigorous and internally consistent, but there is an uncomfortable question:

How much of what we&rsquo;ve learned depends on the particular history we happened to observe?

This question isn&rsquo;t unique to finance. It surfaces anywhere we study systems that cannot be rerun from the same initial conditions. Finance provides an unusually rich laboratory: decades of publicly available data, countless competing models, and no shortage of confident claims.

When reading financial blogs I frequently encounter titles and posts claiming that a new model has &ldquo;beaten the market.&rdquo; In my own research the claims are 'optimized' or 'optimally hedged', but I know the model is brittle, and I have the reoccurring questions in my mind. 

-   Would it still have worked if events unfolded in a different order?
-   Are any of the conclusions robust? Which depend on fortunate sequencing?
-   Are we discovering properties of the market, or properties of one particular realization of history?


## 2. Realization

My curiosity grew about whether these conclusions were actually stable and whether path dependence itself could be quantified. Those questions lingered because trusting (selling) a model without understanding its sensitivity felt incomplete.

Imagine building a bridge. You wouldn&rsquo;t test it under a single gust of wind and immediately label the bridge as &rsquo;safe&rsquo;. You would (hopefully) expose it to thousands of simulated conditions: stronger winds, differing traffic patterns, shifting temperatures, even unlikely combinations of events. The goal isn&rsquo;t to predict the exact forces the bridge will experience. It&rsquo;s to understand how the bridge behaves across many plausible futures.

Most analyses are conditioned on a single realized history. So, why trust one realization so much? Producing a convincing backtest with the typical conclusions: this strategy works, these factors matter, this portfolio is optimally hedged, can certainly be done, but all those statements are conditioned on a single realization of history. Sometimes we can get lucky and look backwards an extra decade or at another economy to evaluate models, but that gets convoluted too.

What if the same underlying market had unfolded a little differently?

Suppose earnings surprises arriving in a different order. A recession delayed by six months. A recovery that began more gradually. The same ingredients, arranged into a different event sequence.

What conclusions survive?

Instead of asking whether a model explains the historical path, I began asking how its conclusions change across plausible historical paths. Stability becomes just as important as accuracy. Model fragility becomes something to investigate rather than simply discover in hindsight.

That realization became the idea of Pathspace Lab.


## 3. Alternative Paths

We only observe one realization of history, yet many questions implicitly concern histories never observed. How stable is an optimized portfolio? Would the same factors emerge? How much of a model&rsquo;s success depends on the particular sequence of events that unfolded?

Traditional workflows offer two directions for answering these questions. We can look backward, fitting models to explain the realized history, or we can look forward, using those models to forecast what might come next.

Both are valuable.

Yet the present was not inevitable. The same market could plausibly have arrived here through a different sequence of events.

That question doesn&rsquo;t require us to predict the future or reinterpret the past. It asks us to explore the neighborhood around the realized history itself.

Rather than asking models to look backward or forward, I wanted a model to look sideways.

That became the central question behind Pathspace Lab.

Sideways means exploring neighboring histories—alternative realizations that remain faithful to what is implied within the data. These histories should preserve the statistical structure captured by a fitted risk model while allowing the sequence of events to change. The objective is not to invent crises that never occurred or predict ones that might. It is to understand how sensitive our conclusions are to the particular ordering of events that we happened to have observe.

To investigate the lateral histories, I began with a conservative and modest experiment: blocked bootstrapping. It introduces no new information, assumes no new market regimes, and changes no model parameters. It simply rearranges pieces of the observed history into alternative, but still plausible, timelines.

If conclusions change dramatically under such small perturbations, then perhaps the realized path was carrying more of the explanation than we realized. If they remain stable, our confidence in those conclusions grows stronger.

Either outcome is informative.

Forecasting tells us where we might go.

Backtesting tells us where we have been.

**Looking sideways asks whether what we believe depends on the particular path we took to get here.**


## 4. What Changes When We Look Sideways?

Changing the direction of questioning changes the kinds of answers we seek.

Instead of asking whether a portfolio achieved the highest historical return, we can ask whether its allocation remains stable across alternative histories.

Instead of accepting a single estimate of risk, we can ask how much that estimate varies when the same market is experienced through different sequences of events.

Instead of reporting a factor exposure, we can ask whether that exposure is an enduring property of the asset or an artifact of the realized path.

The objective is no longer to produce a single estimate. It is to understand the distribution of plausible outcomes surrounding that estimate.

A fragile conclusion is still a conclusion. Discovering where a model is sensitive is just as valuable as knowing where it performs well.

This perspective naturally shifts the emphasis of analysis. Rather than searching for the single &ldquo;best&rdquo; model, we can begin asking which models are robust, which assumptions matter most, and how uncertainty propagates through every stage of the modeling process.

Inference produces probabilities of conclusions. Pathspace asks how those conclusions change when history does.


## 5. Why Pathspace?

Before I worked on financial models, I studied stochastic systems in biophysics.

It came from my background in biophysics, where many systems are understood not through a single trajectory, but through an ensemble of possible ones. A protein folds along one path through its energy landscape, while countless other trajectories remain possible. A single particle tracking trajectory observes one stochastic path, even though many others could have been realized under the same underlying dynamics.

The language of path spaces was familiar there. A path is the sequence of states a system passes through over time. A pathspace is the collection of all possible trajectories that system might take.

As I moved from biophysics into quantitative finance, that intuition remained relevant. In biophysics, it felt natural to think about ensembles of trajectories. In finance, I kept wondering what we might learn if we treated market history in a similar way—not by replacing the realized path, but by studying the space around it.

Pathspace Lab is an attempt to borrow that perspective. The name is a reminder that there are more paths than the realized one.


## 6. What Comes Next?

Once history becomes a neighborhood instead of a single line, entirely new questions become possible.

*How stable is an optimized portfolio when applied to equally plausible histories?*

*Which factor exposures survive across neighboring paths, and which dissolve?*

*How should uncertainty itself be visualized when every model carries an ensemble of plausible histories?*

A single history framework confirms what is already believed. It is far less interesting than one that reveals where our understanding begins to break down.

The underlying questions are not unique to finance. They arise wherever we build models from observations of complex, path-dependent systems.

We only get to observe one path. But understanding it may require exploring the space around it.
