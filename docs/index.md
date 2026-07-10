---
layout: splash
title: "Simulating Alternative Paths"
excerpt: "Because a single backtest is merely a story."
author_profile: false
header:
  overlay_color: "#123a63"
  actions:
    - label: "Featured Project"
      url: /artifact_01_spec.html
intro:
  - title: 'What is a <span class="pathspace-term">pathspace</span>?'
theme_row:
  - title: "Risk Models and Market Structure"
    excerpt: "Understanding how factors, sectors, and residual behavior shape market dynamics."
  - title: "Alternative Histories"
    excerpt: "Generating plausible market paths and stress scenarios from risk decompositions and stochastic processes."
  - title: "Fragility and Uncertainty"
    excerpt: "Studying how systems respond to shocks, regime changes, and rare events."
project_row:
  - image_path: /assets/artifact-01/rolling-weight-bands.png
    alt: "Bootstrap rolling portfolio weight dispersion with realized path overlay"
    title: "Alternative Market Histories - Research Artifact-01"
    excerpt: "A research project exploring how portfolio decisions change when the same market history is resampled into plausible alternative time paths."
    url: /artifact_01_spec.html
    btn_label: "Read Artifact 01"
    btn_class: "btn--primary"
figure_row:
  - image_path: /assets/artifact-01/bootstrap-paths-aapl.png
    alt: "AAPL bootstrap cumulative return paths"
    title: "Bootstrap Paths"
    excerpt: "Example alternative time paths generated from observed log returns."
  - image_path: /assets/artifact-01/cumulative-return-dispersion.png
    alt: "Portfolio cumulative return dispersion across bootstrap paths"
    title: "Return Dispersion"
    excerpt: "Portfolio cumulative return dispersion across bootstrap paths, with the realized path shown as a reference."
  - image_path: /assets/artifact-01/rolling-weight-std.png
    alt: "Cross-path standard deviation of rolling portfolio weights"
    title: "Weight Fragility"
    excerpt: "Cross-path weight dispersion shows when the optimizer becomes most sensitive to the sampled return path."
---

{% include feature_row id="intro" type="center" %}

In mathematics and physics, a **pathspace** is the collection of all possible paths a system could take through time.

Most analysis focuses on the *single path* that occurred.

Pathspace Lab explores the **larger space of paths** that *could have occurred*. By studying alternative histories, simulated futures, and unexpected outcomes, we can better understand *fragility*, *resilience*, and **uncertainty** in complex systems.

{% include feature_row id="theme_row" %}

## Featured Project

{% include feature_row id="project_row" %}

Starting from observed asset and factor returns, Artifact 01 uses block bootstrap sampling to construct alternative histories that preserve local return structure while changing the realized sequence. Each path is passed through the same factor decomposition and rolling minimum-variance optimizer, making it possible to compare the realized path against a distribution of portfolios and outcomes that could have emerged from the same data.

The goal is not prediction.

The goal is exploration.

## About

I'm Drew Tilley, a quantitative researcher and software engineer working at the intersection of risk modeling, simulation, and complex systems.

My background includes developing equity risk models, portfolio optimization tools, and large-scale financial datasets.

Pathspace Lab is my independent research effort to explore uncertainty, alternative histories, and generative approaches to modeling complex systems.

## Current Status

Pathspace Lab is an active research project. Current efforts include block bootstrap time-path generation, portfolio and factor-model research, and open-source tooling.

The project is intentionally exploratory and welcomes discussion and collaboration.

## Figures

{% include feature_row id="figure_row" %}
