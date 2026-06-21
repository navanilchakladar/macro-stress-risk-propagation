# Macro Stress Risk Propagation Framework

A Python framework for propagating macroeconomic stress scenarios into portfolio-level asset shocks using covariance-based risk transmission.

## Overview

This project implements a macro stress testing framework for multi-asset portfolios. The model estimates how shocks to macro and market factors may propagate into portfolio asset sleeves such as equities, bonds, and commodities.

The framework is designed around institutional portfolio risk concepts:

- Predictive stress testing
- Cross-asset covariance transmission
- EWMA covariance estimation
- Macro factor shock propagation
- Portfolio stress contribution
- Weighted R-squared diagnostics
- Mahalanobis stress severity diagnostics


---


## Methodology

The model partitions the covariance matrix into asset and factor blocks:

```text
Σ = [ Σ11  Σ12 ]
    [ Σ21  Σ22 ]

Asset Shock = Σ12 × inverse(Σ22) × Factor Shock

Global Equity Shock:   -10.0%
Global Bond Shock:      -5.0%
Inflation Shock:        +2.0%
Credit Spread Shock:    +2.0%
Commodity Shock:        +5.0%
