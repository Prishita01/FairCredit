# FairCredit — Responsible AI for Credit Risk Assessment

> 68.7% fairness improvement with only 2.3% accuracy loss. Validated on 3 datasets.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Does

End-to-end responsible AI pipeline for credit risk scoring covering fairness auditing, bias mitigation, robustness testing, and SHAP
explainability, validated across three real-world credit datasets.

**Core finding:** Removing sensitive attributes like Sex and Age does NOT eliminate bias. SHAP group-wise analysis shows bias flows through proxy variables — employment status, credit history, credit amount, which encode group information indirectly. Fairness-aware training is non-negotiable.

---

## Results

| Metric | Baseline (RF) | After Mitigation |
|---|---|---|
| Accuracy | 0.757 | 0.735 (-2.2% only) |
| AUC-ROC | 0.809 | 0.808 |
| EOD (Sex) | 0.128 | 0.013 (**90% reduction**) |
| DPD (Sex) | 0.165 | 0.052 (**68.5% reduction**) |
| DI (Sex) | 0.751 | 0.862 (**meets ≥0.8 compliance**) |

Trade-off ratio: **31:1** — for every 1% accuracy lost, fairness improved by 31%.

Validated on:
- German Credit Dataset — 1,000 instances
- Portuguese Bank Marketing — 45,211 instances  
- Synthetic US Credit — 10,000 instances

---

## Mitigation Approaches

| Method | Type | Accuracy Loss | Fairness Gain |
|---|---|---|---|
| Reweighing | Pre-processing | -1.4% | 50.4% |
| Threshold Optimization | Post-processing | -2.2% | **68.4%** |

---

## Robustness

Model was stress-tested under:
- Covariate shift — accuracy drops up to 15%
- Label shift — moderate degradation
- MNAR missingness — strongest degradation
- Gaussian noise — stable at low-moderate levels

Fair models on static test sets are not guaranteed under
distribution shift. Production monitoring is required.

---

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

The system uses YAML configuration files for experiment management. See `config/default_config.yaml` for the default configuration.

## Usage

```python
from fair_credit import Config, FairCreditPipeline

# Load configuration
config = Config.from_file('config/default_config.yaml')

# Initialize pipeline
pipeline = FairCreditPipeline(config)

# Run full pipeline
results = pipeline.run_full_pipeline()
```

## License

This project is developed for research and educational purposes.

Note: Core fairness auditing and mitigation modules are included.

Tech Stack
Python · PyTorch · Scikit-learn · AIF360 · SHAP · Pandas · NumPy


