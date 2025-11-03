# Task 6: Post-Processing Bias Mitigation - Mathematical Explanation

## Executive Summary

Task 6 implements post-processing bias mitigation through threshold optimization, a mathematically principled approach that adjusts decision thresholds for different demographic groups to achieve fairness while preserving model utility. This document provides a comprehensive mathematical explanation of the three subtasks and their theoretical foundations.

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Subtask 6.1: Threshold Optimization Algorithm](#2-subtask-61-threshold-optimization-algorithm)
3. [Subtask 6.2: Threshold Application System](#3-subtask-62-threshold-application-system)
4. [Subtask 6.3: Effectiveness Evaluation](#4-subtask-63-effectiveness-evaluation)
5. [Theoretical Guarantees](#5-theoretical-guarantees)
6. [Computational Complexity](#6-computational-complexity)
7. [Comparison with Pre-Processing](#7-comparison-with-pre-processing)

---

## 1. Mathematical Foundation

### 1.1 Problem Setup

Let us define the mathematical framework for post-processing bias mitigation:

**Given:**
- Feature space: **X** ∈ ℝᵈ
- Binary labels: **Y** ∈ {0, 1}
- Protected attributes: **A** ∈ {0, 1, ..., k-1} representing k demographic groups
- Trained classifier: **f: X → [0,1]** outputting P(Y=1|X)
- Dataset: **D** = {(xᵢ, yᵢ, aᵢ)}ᵢ₌₁ⁿ

**Objective:** Find group-specific thresholds **τ** = {τ₀, τ₁, ..., τₖ₋₁} such that:

```
Ŷ = 1 if f(X) ≥ τₐ, 0 otherwise
```

where **a** is the protected attribute value for the instance.

### 1.2 Fairness Constraints

#### Equal Opportunity Constraint
For all groups a, b ∈ A:
```
|TPR_a - TPR_b| ≤ ε
```

Where:
```
TPR_a = P(Ŷ=1 | Y=1, A=a) = P(f(X) ≥ τₐ | Y=1, A=a)
```

#### Equalized Odds Constraint
For all groups a, b ∈ A:
```
|TPR_a - TPR_b| ≤ ε  AND  |FPR_a - FPR_b| ≤ ε
```

Where:
```
FPR_a = P(Ŷ=1 | Y=0, A=a) = P(f(X) ≥ τₐ | Y=0, A=a)
```

### 1.3 Optimization Formulation

The complete optimization problem is:

```
minimize    L(τ) = (1/n) Σᵢ₌₁ⁿ ℓ(yᵢ, ŷᵢ(τ))
subject to  g_fairness(τ) ≤ ε
           0 ≤ τₐ ≤ 1  ∀a ∈ A
```

Where:
- **ℓ(y, ŷ)** is the loss function (typically 0-1 loss)
- **ŷᵢ(τ) = 𝟙[f(xᵢ) ≥ τₐᵢ]** is the prediction for instance i
- **g_fairness(τ)** represents fairness constraint violations

---

## 2. Subtask 6.1: Threshold Optimization Algorithm

### 2.1 Constrained Optimization Problem

**Formal Statement:**
```
minimize    E[L(Y, Ŷ)] = Σₐ P(A=a) · E[L(Y, Ŷ) | A=a]
subject to  max_{a,b} |Metric_a(τ) - Metric_b(τ)| ≤ ε
           τₐ ∈ [0, 1]  ∀a
```

**Discrete Approximation:**
For finite sample size n:
```
minimize    (1/n) Σᵢ₌₁ⁿ L(yᵢ, ŷᵢ)
subject to  max_{a,b} |TPR_a - TPR_b| ≤ ε
           0 ≤ τₐ ≤ 1  ∀a
```

### 2.2 Solution Algorithms

#### 2.2.1 Differential Evolution

**Algorithm Overview:**
Differential Evolution (DE) is a population-based global optimization algorithm particularly suited for non-convex, non-differentiable problems.

**Mathematical Formulation:**

1. **Population Initialization:**
   ```
   τ⁽⁰⁾ᵢ ~ U(0, 1)ᵏ  for i = 1, ..., NP
   ```
   where NP is the population size.

2. **Mutation Operation:**
   ```
   vᵢ⁽ᵍ⁺¹⁾ = τᵣ₁⁽ᵍ⁾ + F · (τᵣ₂⁽ᵍ⁾ - τᵣ₃⁽ᵍ⁾)
   ```
   where F ∈ (0, 2] is the mutation factor, and r₁, r₂, r₃ are distinct random indices.

3. **Crossover Operation:**
   ```
   uᵢⱼ⁽ᵍ⁺¹⁾ = {
     vᵢⱼ⁽ᵍ⁺¹⁾  if rand(0,1) ≤ CR or j = jᵣₐₙ𝒹
     τᵢⱼ⁽ᵍ⁾    otherwise
   }
   ```
   where CR ∈ [0, 1] is the crossover probability.

4. **Selection Operation:**
   ```
   τᵢ⁽ᵍ⁺¹⁾ = {
     uᵢ⁽ᵍ⁺¹⁾  if f(uᵢ⁽ᵍ⁺¹⁾) ≤ f(τᵢ⁽ᵍ⁾)
     τᵢ⁽ᵍ⁾    otherwise
   }
   ```

#### 2.2.2 Sequential Least Squares Programming (SLSQP)

**Lagrangian Formulation:**
```
L(τ, λ, μ) = f(τ) + Σⱼ λⱼ gⱼ(τ) + Σₖ μₖ hₖ(τ)
```

**KKT Conditions:**
At the optimal solution τ*, there exist multipliers λ*, μ* such that:
```
∇f(τ*) + Σⱼ λⱼ* ∇gⱼ(τ*) + Σₖ μₖ* ∇hₖ(τ*) = 0
λⱼ* gⱼ(τ*) = 0  ∀j  (complementary slackness)
λⱼ* ≥ 0  ∀j
gⱼ(τ*) ≤ 0  ∀j
hₖ(τ*) = 0  ∀k
```

### 2.3 Constraint Functions

#### Equal Opportunity Constraint Function
```
g_EO(τ) = max_{a,b} |TPR_a(τ) - TPR_b(τ)| - ε
```

Where:
```
TPR_a(τ) = Σᵢ∈{Y=1,A=a} 𝟙[f(xᵢ) ≥ τₐ] / |{i: yᵢ=1, aᵢ=a}|
```

#### Equalized Odds Constraint Function
```
g_EqOdds(τ) = max(max_{a,b} |TPR_a(τ) - TPR_b(τ)|, max_{a,b} |FPR_a(τ) - FPR_b(τ)|) - ε
```

Where:
```
FPR_a(τ) = Σᵢ∈{Y=0,A=a} 𝟙[f(xᵢ) ≥ τₐ] / |{i: yᵢ=0, aᵢ=a}|
```

### 2.4 Convergence Analysis

**Theorem 1 (Convergence of DE):** Under mild conditions on the objective function, the differential evolution algorithm converges to the global optimum with probability 1 as the number of generations approaches infinity.

**Proof Sketch:** The proof relies on the fact that DE maintains diversity in the population and has a non-zero probability of generating any point in the search space at each generation.

---

## 3. Subtask 6.2: Threshold Application System

### 3.1 Validation Set Methodology

**Data Splitting:**
```
D = D_train ∪ D_val
|D_val| = α · |D|  where α ∈ (0, 1)
```

**Stratified Splitting:**
Ensure proportional representation of groups and labels:
```
P(A=a, Y=y | D_train) ≈ P(A=a, Y=y | D_val) ≈ P(A=a, Y=y | D)
```

### 3.2 Overfitting Detection

**Mathematical Formulation:**

Define the generalization gap for metric M:
```
Gap_M = |M(D_train, τ*) - M(D_val, τ*)|
```

**Overfitting Indicators:**
```
Overfitting = (Gap_Fairness > δ_f) ∨ (Gap_Accuracy > δ_a)
```

Where δ_f and δ_a are predefined thresholds (typically 0.05).

### 3.3 Decision Boundary Analysis

**Threshold Impact Function:**
For group a, the decision boundary in probability space is:
```
B_a(τ_a) = {x ∈ X : f(x) = τ_a}
```

**Near-Threshold Region:**
```
N_a(τ_a, δ) = {x ∈ X : |f(x) - τ_a| ≤ δ}
```

**Consistency Verification:**
```
Consistency_a = ∀i ∈ {A_i = a}: (f(x_i) ≥ τ_a) ⟺ (ŷ_i = 1)
```

### 3.4 Validation Metrics

**Cross-Validation Error:**
```
CV_Error = (1/K) Σₖ₌₁ᴷ L(D_val^(k), τ*_train^(k))
```

**Stability Measure:**
```
Stability = Var_k[τ*_train^(k)]
```

Where τ*_train^(k) are optimal thresholds computed on different training folds.

---

## 4. Subtask 6.3: Effectiveness Evaluation

### 4.1 Fairness Improvement Metrics

**Absolute Improvement:**
```
Δ_abs^M = M_baseline - M_mitigated
```

**Relative Improvement:**
```
Δ_rel^M = (M_baseline - M_mitigated) / M_baseline
```

**Success Criterion:**
```
Success_Fairness = Δ_rel^EO ≥ 0.5  (≥50% improvement)
```

### 4.2 Utility Preservation Metrics

**AUC Preservation Property:**
```
AUC_post = AUC_baseline  (exactly, due to probability preservation)
```

**Accuracy Change:**
```
Δ_Accuracy = Accuracy_mitigated - Accuracy_baseline
```

**Success Criterion:**
```
Success_Utility = Δ_Accuracy ≥ -0.02  (≤2% degradation)
```

### 4.3 Comparative Analysis Framework

**Multi-Objective Evaluation:**
```
Score(method) = w₁ · Fairness_Improvement + w₂ · Utility_Preservation
```

**Pareto Dominance:**
Method A dominates method B if:
```
Fairness_A ≥ Fairness_B  AND  Utility_A ≥ Utility_B
```
with at least one strict inequality.

### 4.4 Statistical Significance Testing

**Hypothesis Testing:**
```
H₀: Δ_rel^EO ≤ 0  (no improvement)
H₁: Δ_rel^EO > 0  (improvement)
```

**Test Statistic:**
```
t = (Δ̂_rel^EO - 0) / SE(Δ̂_rel^EO)
```

Where SE is computed using bootstrap or analytical methods.

---

## 5. Theoretical Guarantees

### 5.1 Fairness Guarantees

**Theorem 2 (Constraint Satisfaction):** If the optimization algorithm converges to a feasible solution τ*, then:
```
|TPR_a(τ*) - TPR_b(τ*)| ≤ ε  ∀a,b ∈ A
```

**Proof:** By construction of the constraint set and the definition of feasibility.

### 5.2 Utility Bounds

**Theorem 3 (AUC Preservation):** Post-processing cannot change the AUC:
```
AUC_post = AUC_baseline
```

**Proof:** AUC depends only on the ranking of probabilities, which are unchanged in post-processing. Formally:
```
AUC = P(f(X₁) > f(X₀) | Y₁=1, Y₀=0)
```
Since f(X) is unchanged, the ranking is preserved.

**Theorem 4 (Accuracy Bounds):** The accuracy after post-processing is bounded by:
```
Accuracy_post ≤ max_τ Accuracy(τ)
```

Where the maximum is over all possible threshold configurations.

### 5.3 Optimality Conditions

**Theorem 5 (First-Order Optimality):** At the optimal solution τ*, the following conditions hold:

1. **Stationarity:** ∇L(τ*) + Σⱼ λⱼ* ∇gⱼ(τ*) = 0
2. **Primal Feasibility:** gⱼ(τ*) ≤ 0  ∀j
3. **Dual Feasibility:** λⱼ* ≥ 0  ∀j
4. **Complementary Slackness:** λⱼ* gⱼ(τ*) = 0  ∀j

---

## 6. Computational Complexity

### 6.1 Time Complexity

**Threshold Optimization:**
- **Differential Evolution:** O(G · NP · n · k)
- **SLSQP:** O(I · n · k²)

Where:
- G = number of generations
- NP = population size
- I = number of iterations
- n = dataset size
- k = number of groups

**Threshold Application:**
- **Single Prediction:** O(1)
- **Batch Prediction:** O(m) for m instances

### 6.2 Space Complexity

**Memory Requirements:**
- **Threshold Storage:** O(k)
- **Optimization State:** O(NP · k) for DE, O(k²) for SLSQP
- **Data Storage:** O(n · d) where d is feature dimensionality

### 6.3 Scalability Analysis

**Scaling with Dataset Size:**
```
T(n) = O(n)  (linear scaling)
```

**Scaling with Number of Groups:**
```
T(k) = O(k²)  (quadratic due to pairwise constraints)
```

---

## 7. Comparison with Pre-Processing

### 7.1 Mathematical Differences

| Aspect | Pre-Processing | Post-Processing |
|--------|----------------|-----------------|
| **Optimization Variable** | Sample weights w(a,y) | Thresholds τₐ |
| **Model Change** | ✓ (retraining required) | ✗ (model fixed) |
| **Probability Change** | ✓ (f_new ≠ f_old) | ✗ (f unchanged) |
| **Decision Change** | ✓ | ✓ |

### 7.2 Theoretical Properties

**Pre-Processing (Reweighing):**
```
w(a,y) = P(A=a) · P(Y=y) / P(A=a, Y=y)
```

**Objective:** Achieve statistical independence P(A ⊥ Y | w)

**Post-Processing (Threshold Optimization):**
```
τₐ = argmin L(τ) subject to fairness constraints
```

**Objective:** Minimize classification error subject to fairness

### 7.3 Convergence Properties

**Pre-Processing Convergence:**
- Depends on model training convergence
- May not converge to global optimum due to non-convex loss

**Post-Processing Convergence:**
- Guaranteed convergence for convex constraints
- Global optimization possible with DE

### 7.4 Robustness Analysis

**Sensitivity to Data Distribution:**
- **Pre-Processing:** High sensitivity (requires retraining)
- **Post-Processing:** Lower sensitivity (threshold adjustment)

**Computational Stability:**
- **Pre-Processing:** Depends on model stability
- **Post-Processing:** More stable (simpler optimization)

---

## 8. Implementation Considerations

### 8.1 Numerical Stability

**Threshold Bounds:**
```
τₐ ∈ [ε, 1-ε]  where ε > 0 is small
```

**Constraint Tolerance:**
```
|Metric_a - Metric_b| ≤ ε + δ_numerical
```

### 8.2 Hyperparameter Selection

**Optimization Parameters:**
- **DE:** F ∈ [0.5, 1.0], CR ∈ [0.7, 0.9], NP = 10k
- **SLSQP:** ftol = 1e-9, maxiter = 1000

**Validation Parameters:**
- **Split ratio:** α ∈ [0.2, 0.3]
- **Overfitting threshold:** δ = 0.05

### 8.3 Practical Guidelines

**When to Use Post-Processing:**
1. Model retraining is expensive: O(model_training) >> O(threshold_optimization)
2. Regulatory requirements change frequently
3. Need to preserve model calibration
4. Working with third-party models

**Success Criteria:**
1. **Fairness:** Δ_rel^EO ≥ 0.5
2. **Utility:** Δ_Accuracy ≥ -0.02
3. **Constraint:** Actual_gap ≤ ε
4. **Stability:** Low variance across validation folds

---

## 9. Conclusion

Task 6 implements a mathematically rigorous post-processing bias mitigation framework with the following key contributions:

1. **Constrained Optimization (6.1):** Formulates fairness as a constrained optimization problem with theoretical guarantees on convergence and optimality.

2. **Validation Framework (6.2):** Implements proper validation methodology to prevent overfitting with mathematical bounds on generalization error.

3. **Comparative Analysis (6.3):** Provides comprehensive evaluation framework with statistical significance testing and multi-objective optimization.

The mathematical foundation ensures that:
- **Fairness constraints are satisfied** within specified tolerance
- **Utility is preserved** with bounded degradation
- **Computational efficiency** is maintained through linear scaling
- **Theoretical guarantees** provide confidence in the approach

This framework provides a principled, mathematically sound approach to achieving fairness in machine learning systems while maintaining practical deployability and theoretical rigor.
