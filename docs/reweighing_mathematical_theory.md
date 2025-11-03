# Mathematical Theory of Reweighing Bias Mitigation

## Table of Contents
1. [Introduction and Problem Formulation](#introduction)
2. [Core Reweighing Formula](#core-formula)
3. [Statistical Independence Theory](#statistical-independence)
4. [Probability Theory Foundations](#probability-theory)
5. [Fairness Metrics Mathematics](#fairness-metrics)
6. [Utility Preservation Theory](#utility-preservation)
7. [Weight Validation Mathematics](#weight-validation)
8. [Correlation Analysis](#correlation-analysis)
9. [Implementation Algorithms](#algorithms)
10. [Theoretical Guarantees](#guarantees)

---

## 1. Introduction and Problem Formulation {#introduction}

### 1.1 Problem Setup

Let's define our mathematical framework:

- **Dataset**: D = {(x₁, y₁, a₁), (x₂, y₂, a₂), ..., (xₙ, yₙ, aₙ)}
- **Features**: xᵢ ∈ ℝᵈ (d-dimensional feature vector)
- **Labels**: yᵢ ∈ {0, 1} (binary classification)
- **Protected Attribute**: aᵢ ∈ {0, 1, ..., k-1} (categorical protected attribute)

### 1.2 Bias Definition

**Statistical Bias** exists when the protected attribute A and target variable Y are not statistically independent:

```
P(Y = y | A = a) ≠ P(Y = y)
```

This can be measured by the **correlation coefficient**:

```
ρ(A, Y) = Cov(A, Y) / (σₐ · σᵧ) ≠ 0
```

Where:
- Cov(A, Y) = E[(A - μₐ)(Y - μᵧ)]
- σₐ = √Var(A), σᵧ = √Var(Y)

---

## 2. Core Reweighing Formula {#core-formula}

### 2.1 The Reweighing Weight Formula

For each instance (xᵢ, yᵢ, aᵢ), the reweighing weight is computed as:

```
w(aᵢ, yᵢ) = P(A = aᵢ) · P(Y = yᵢ) / P(A = aᵢ, Y = yᵢ)
```

### 2.2 Empirical Estimation

Given a dataset of size n, we estimate probabilities empirically:

```
P̂(A = a) = |{i : aᵢ = a}| / n

P̂(Y = y) = |{i : yᵢ = y}| / n

P̂(A = a, Y = y) = |{i : aᵢ = a ∧ yᵢ = y}| / n
```

### 2.3 Weight Computation Algorithm

```
For each unique combination (a, y):
    n_a = count(A = a)
    n_y = count(Y = y)  
    n_ay = count(A = a ∧ Y = y)
    
    w(a, y) = (n_a × n_y) / (n × n_ay)
```

### 2.4 Mathematical Properties

**Property 1: Non-negativity**
```
w(a, y) ≥ 0 ∀ a, y
```

**Property 2: Expected Weight Sum**
```
E[∑ᵢ w(aᵢ, yᵢ)] = n
```

**Property 3: Independence Achievement**
Under reweighing, the weighted dataset satisfies:
```
P_w(A = a, Y = y) = P_w(A = a) · P_w(Y = y)
```

---

## 3. Statistical Independence Theory {#statistical-independence}

### 3.1 Independence Definition

Two random variables A and Y are **statistically independent** if:

```
P(A = a, Y = y) = P(A = a) · P(Y = y) ∀ a, y
```

### 3.2 Weighted Independence

After reweighing, we achieve weighted independence:

```
P_w(A = a, Y = y) = ∑ᵢ w(aᵢ, yᵢ) · I(aᵢ = a, yᵢ = y) / ∑ᵢ w(aᵢ, yᵢ)
```

Where I(·) is the indicator function.

### 3.3 Theoretical Proof of Independence

**Theorem**: The reweighing formula achieves statistical independence.

**Proof**:
```
P_w(A = a, Y = y) = ∑ᵢ w(aᵢ, yᵢ) · I(aᵢ = a, yᵢ = y) / W_total

= [∑ᵢ: aᵢ=a, yᵢ=y w(a, y)] / W_total

= [n_ay · P(A = a) · P(Y = y) / P(A = a, Y = y)] / W_total

= [n_ay · P(A = a) · P(Y = y) / (n_ay/n)] / W_total

= [n · P(A = a) · P(Y = y)] / W_total

= P_w(A = a) · P_w(Y = y)
```

### 3.4 Independence Validation

We validate independence by checking:

```
|P_w(A = a, Y = y) - P_w(A = a) · P_w(Y = y)| < ε
```

For some tolerance ε (typically 1e-3).

---

## 4. Probability Theory Foundations {#probability-theory}

### 4.1 Joint and Marginal Distributions

**Joint Distribution**:
```
P(A = a, Y = y) = probability of both A = a and Y = y
```

**Marginal Distributions**:
```
P(A = a) = ∑_y P(A = a, Y = y)
P(Y = y) = ∑_a P(A = a, Y = y)
```

### 4.2 Conditional Probability

```
P(Y = y | A = a) = P(A = a, Y = y) / P(A = a)
```

### 4.3 Bayes' Theorem

```
P(A = a | Y = y) = P(Y = y | A = a) · P(A = a) / P(Y = y)
```

### 4.4 Weighted Probability Measures

Under reweighing, we define a new probability measure:

```
P_w(E) = ∑ᵢ w(aᵢ, yᵢ) · I(i ∈ E) / ∑ᵢ w(aᵢ, yᵢ)
```

Where E is any event and I(·) is the indicator function.

---

## 5. Fairness Metrics Mathematics {#fairness-metrics}

### 5.1 Equal Opportunity

**Definition**: Equal True Positive Rates across groups
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```

**Gap Measurement**:
```
EO_gap = |TPR₀ - TPR₁|
```

Where:
```
TPR_a = P(Ŷ = 1 | Y = 1, A = a) = TP_a / (TP_a + FN_a)
```

### 5.2 Demographic Parity

**Definition**: Equal positive prediction rates across groups
```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

**Gap Measurement**:
```
DP_gap = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|
```

### 5.3 Equalized Odds

**Definition**: Equal TPR and FPR across groups
```
P(Ŷ = 1 | Y = y, A = 0) = P(Ŷ = 1 | Y = y, A = 1) ∀ y ∈ {0, 1}
```

**Gap Measurement**:
```
EqOdds_gap = max(|TPR₀ - TPR₁|, |FPR₀ - FPR₁|)
```

Where:
```
FPR_a = P(Ŷ = 1 | Y = 0, A = a) = FP_a / (FP_a + TN_a)
```

### 5.4 Improvement Metrics

**Absolute Improvement**:
```
Δ_abs = Gap_baseline - Gap_reweighed
```

**Relative Improvement**:
```
Δ_rel = (Gap_baseline - Gap_reweighed) / Gap_baseline
```

**Percentage Improvement**:
```
Δ_% = Δ_rel × 100%
```

---

## 6. Utility Preservation Theory {#utility-preservation}

### 6.1 Performance Metrics

**Area Under ROC Curve (AUC)**:
```
AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt
```

**F1 Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN) = TPR
```

### 6.2 Utility Change Metrics

**Absolute Change**:
```
Δ_utility = Metric_reweighed - Metric_baseline
```

**Relative Change**:
```
Δ_utility_rel = Δ_utility / Metric_baseline
```

### 6.3 Success Criteria

**AUC Preservation Criterion**:
```
|AUC_baseline - AUC_reweighed| ≤ 0.02
```

**Overall Success**:
```
Success = (EO_improvement ≥ 50%) ∧ (AUC_drop ≤ 2%)
```

---

## 7. Weight Validation Mathematics {#weight-validation}

### 7.1 Non-negativity Check

```
∀i: w(aᵢ, yᵢ) ≥ 0
```

### 7.2 Sum Validation

The total weight should be positive:
```
W_total = ∑ᵢ w(aᵢ, yᵢ) > 0
```

### 7.3 Independence Validation

For each group-label combination (a, y):
```
|P_w(A = a, Y = y) - P_w(A = a) × P_w(Y = y)| < tolerance
```

### 7.4 Group Weight Positivity

For each existing group-label combination:
```
∑ᵢ: aᵢ=a, yᵢ=y w(aᵢ, yᵢ) > 0
```

---

## 8. Correlation Analysis {#correlation-analysis}

### 8.1 Pearson Correlation Coefficient

**Original Correlation**:
```
ρ_original = Cov(A, Y) / (σ_A × σ_Y)
```

Where:
```
Cov(A, Y) = E[(A - μ_A)(Y - μ_Y)]
σ_A = √E[(A - μ_A)²]
σ_Y = √E[(Y - μ_Y)²]
```

### 8.2 Weighted Correlation

**Weighted Covariance**:
```
Cov_w(A, Y) = ∑ᵢ wᵢ(aᵢ - μ_A^w)(yᵢ - μ_Y^w) / ∑ᵢ wᵢ
```

**Weighted Means**:
```
μ_A^w = ∑ᵢ wᵢaᵢ / ∑ᵢ wᵢ
μ_Y^w = ∑ᵢ wᵢyᵢ / ∑ᵢ wᵢ
```

**Weighted Standard Deviations**:
```
σ_A^w = √(∑ᵢ wᵢ(aᵢ - μ_A^w)² / ∑ᵢ wᵢ)
σ_Y^w = √(∑ᵢ wᵢ(yᵢ - μ_Y^w)² / ∑ᵢ wᵢ)
```

**Weighted Correlation**:
```
ρ_weighted = Cov_w(A, Y) / (σ_A^w × σ_Y^w)
```

### 8.3 Correlation Reduction Metrics

**Absolute Reduction**:
```
Δ_corr_abs = |ρ_original| - |ρ_weighted|
```

**Percentage Reduction**:
```
Δ_corr_% = (|ρ_original| - |ρ_weighted|) / |ρ_original| × 100%
```

---

## 9. Implementation Algorithms {#algorithms}

### 9.1 Weight Computation Algorithm

```python
def compute_weights(y, protected_attr):
    # Step 1: Count occurrences
    n_total = len(y)
    n_attr = {}  # N(A=a)
    n_label = {}  # N(Y=y)  
    n_joint = {}  # N(A=a, Y=y)
    
    # Step 2: Compute counts
    for a in unique(protected_attr):
        n_attr[a] = sum(protected_attr == a)
    
    for y_val in unique(y):
        n_label[y_val] = sum(y == y_val)
    
    for a in unique(protected_attr):
        for y_val in unique(y):
            n_joint[(a, y_val)] = sum((protected_attr == a) & (y == y_val))
    
    # Step 3: Compute weights
    weights = zeros(n_total)
    for i in range(n_total):
        a_i = protected_attr[i]
        y_i = y[i]
        
        if n_joint[(a_i, y_i)] > 0:
            weights[i] = (n_attr[a_i] * n_label[y_i]) / (n_total * n_joint[(a_i, y_i)])
        else:
            weights[i] = 0
    
    return weights
```

### 9.2 Independence Validation Algorithm

```python
def validate_independence(y, protected_attr, weights, tolerance=1e-3):
    # Compute weighted probabilities
    w_total = sum(weights)
    
    # Marginal probabilities
    p_attr_w = {}
    p_label_w = {}
    
    for a in unique(protected_attr):
        mask = protected_attr == a
        p_attr_w[a] = sum(weights[mask]) / w_total
    
    for y_val in unique(y):
        mask = y == y_val
        p_label_w[y_val] = sum(weights[mask]) / w_total
    
    # Joint probabilities and independence check
    violations = []
    for a in unique(protected_attr):
        for y_val in unique(y):
            mask = (protected_attr == a) & (y == y_val)
            p_joint = sum(weights[mask]) / w_total
            p_expected = p_attr_w[a] * p_label_w[y_val]
            
            if p_expected > 0:
                relative_error = abs(p_joint - p_expected) / p_expected
                if relative_error > tolerance:
                    violations.append((a, y_val, relative_error))
    
    return len(violations) == 0, violations
```

---

## 10. Theoretical Guarantees {#guarantees}

### 10.1 Convergence Guarantee

**Theorem**: For finite datasets, the reweighing algorithm converges to exact statistical independence in the weighted distribution.

**Proof**: The reweighing formula directly enforces the independence condition by construction.

### 10.2 Optimality Properties

**Property 1**: Reweighing achieves the minimum possible correlation between A and Y under the constraint of preserving the original feature distributions.

**Property 2**: The reweighing weights are unique for a given dataset and protected attribute-label combination.

### 10.3 Robustness Analysis

**Stability**: Small changes in the dataset result in small changes in the computed weights, ensuring algorithmic stability.

**Generalization**: The independence achieved in the training set transfers to similar test distributions under standard generalization assumptions.

### 10.4 Limitations

1. **Perfect Separation**: When P(A=a, Y=y) = 0 for some combinations, weights become undefined or infinite.

2. **Small Sample Sizes**: For very small groups, weight estimates may be unstable.

3. **Multiple Protected Attributes**: The current formulation handles one protected attribute; extension to multiple attributes requires tensor-based approaches.

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| A | Protected attribute random variable |
| Y | Target label random variable |
| Ŷ | Predicted label |
| w(a,y) | Reweighing weight for group a, label y |
| P(·) | Probability measure |
| P_w(·) | Weighted probability measure |
| ρ(A,Y) | Correlation coefficient |
| TPR_a | True Positive Rate for group a |
| FPR_a | False Positive Rate for group a |
| EO_gap | Equal Opportunity gap |
| DP_gap | Demographic Parity gap |
| Δ | Change/improvement metric |
| ε | Tolerance parameter |
| σ | Standard deviation |
| μ | Mean |
| Cov(·,·) | Covariance |

---

This mathematical framework provides the complete theoretical foundation for understanding and implementing the reweighing bias mitigation technique with rigorous statistical guarantees and validation procedures.