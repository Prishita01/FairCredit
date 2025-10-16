# FairCredit: Responsible AI Analysis Pipeline

A comprehensive fairness audit and mitigation pipeline for credit risk scoring using the German Credit dataset.

## Overview

FairCredit implements a modular framework for:
- Baseline model training (Logistic Regression, XGBoost)
- Fairness auditing with Equal Opportunity metrics
- Bias mitigation through reweighing and threshold optimization
- Explainability analysis with SHAP
- Robustness testing under distribution shifts

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

## Requirements

This implementation addresses the following requirements:
- **Requirement 1.1**: German Credit dataset preprocessing with 1,000 instances
- **Requirement 1.3**: Stratified train/validation/test splits by label and protected groups

## Next Steps

The core interfaces and project structure are now established. Subsequent tasks will implement:
1. Data processing pipeline (Task 2)
2. Baseline model training (Task 3)
3. Fairness auditing system (Task 4)
4. Bias mitigation techniques (Tasks 5-6)
5. Explainability analysis (Task 7)
6. Robustness testing (Task 8)
7. Success criteria validation (Task 9)
8. Reporting and visualization (Task 10)
9. End-to-end integration (Task 11)
10. Testing and validation (Task 12)

## License

This project is developed for research and educational purposes.
