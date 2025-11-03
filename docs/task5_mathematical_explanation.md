# Task 5: Mathematical Explanation of Reweighing Bias Mitigation

## Overview: What Are We Solving?

In Task 5, we're addressing **statistical bias** in machine learning datasets where the protected attribute (e.g., gender, race) is correlated with the target variable (e.g., loan approval). This creates unfair models that discriminate against certain groups.

**The Problem**: Given a biased dataset, train a fair model that makes decisions independent of protected attributes while maintaining predictive performance.

**Our Solution**: Reweighing - a pre-processing technique that assigns weights to training instances to decorrelate protected attributes from labels.

---

## Step-by-Step Mathematical Process

### Step 1: Problem Identification

**Input Dataset**: D = {(x₁, y₁, a₁), (x₂, y₂, a₂), ..., (xₙ, yₙ, aₙ)}

Where:
- xᵢ = feature vector for instance i
- yᵢ ∈ {0, 1} = binary label (e.g., 0=reject loan, 1=approve loan)  
- aᵢ ∈ {0, 1} = protected attribute (e.g., 0=female, 1=male)

**Bias Detection**: We measure bias using correlation:
```
ρ(A, Y) = Cov(A, Y) / (σₐ · σᵧ)
```

If ρ(A, Y) ≠ 0, the dataset is biased.

**Example**: 
- Males get 70% loan approvals
- Females get 30% loan approvals
- This creates ρ(gender, approval) = 0.4 (strong positive correlation)

### Step 2: Reweighing Weight Computation

**Core Formula**: For each instance (xᵢ, yᵢ, aᵢ), compute weight:

```
w(aᵢ, yᵢ) = P(A = aᵢ) × P(Y = yᵢ) / P(A = aᵢ, Y = yᵢ)
```

**Intuitive Explanation**: 
- Numerator: What we expect if A and Y were independent
- Denominator: What we actually observe
- Ratio: Correction factor to achieve independence

**Empirical Calculation**:
```
P̂(A = a) = count(A = a) / n
P̂(Y = y) = count(Y = y) / n  
P̂(A = a, Y = y) = count(A = a ∧ Y = y) / n

w(a, y) = [count(A = a) × count(Y = y)] / [n × count(A = a ∧ Y = y)]
```

**Concrete Example**:
Dataset: 1000 instances, 500 male/500 female, 600 approvals/400 rejections

Original distribution:
- Male + Approved: 400 instances
- Male + Rejected: 100 instances  
- Female + Approved: 200 instances
- Female + Rejected: 300 instances

Weights calculation:
```
w(Male, Approved) = (500 × 600) / (1000 × 400) = 0.75
w(Male, Rejected) = (500 × 400) / (1000 × 100) = 2.00
w(Female, Approved) = (500 × 600) / (1000 × 200) = 1.50
w(Female, Rejected) = (500 × 400) / (1000 × 300) = 0.67
```

**Effect**: 
- Male approvals get downweighted (0.75 < 1)
- Male rejections get upweighted (2.00 > 1)
- Female approvals get upweighted (1.50 > 1)
- Female rejections get downweighted (0.67 < 1)

### Step 3: Statistical Independence Achievement

**Goal**: Make P(A = a, Y = y) = P(A = a) × P(Y = y) in the weighted dataset

**Weighted Probability**: 
```
P_w(A = a, Y = y) = Σᵢ w(aᵢ, yᵢ) × I(aᵢ = a, yᵢ = y) / Σᵢ w(aᵢ, yᵢ)
```

**Mathematical Proof of Independence**:
```
P_w(A = a, Y = y) = [n_{a,y} × P(A = a) × P(Y = y) / P(A = a, Y = y)] / W_total
                  = [n_{a,y} × P(A = a) × P(Y = y) / (n_{a,y}/n)] / W_total
                  = [n × P(A = a) × P(Y = y)] / W_total
                  = P_w(A = a) × P_w(Y = y)
```

**Result**: The weighted dataset has ρ_weighted(A, Y) ≈ 0

### Step 4: Model Training with Weights

**Weighted Loss Function**: Instead of standard loss L(θ), we minimize:
```
L_weighted(θ) = Σᵢ w(aᵢ, yᵢ) × ℓ(f(xᵢ; θ), yᵢ)
```

Where:
- f(xᵢ; θ) = model prediction
- ℓ(·, ·) = loss function (e.g., log loss)
- w(aᵢ, yᵢ) = reweighing weight

**Effect**: The model learns from a "rebalanced" version of the data where bias is removed.

### Step 5: Fairness Evaluation

**Equal Opportunity Measurement**:
```
EO_gap = |TPR_male - TPR_female|
```

Where TPR_group = True Positive Rate for each group:
```
TPR_group = P(Ŷ = 1 | Y = 1, A = group) = TP_group / (TP_group + FN_group)
```

**Improvement Calculation**:
```
EO_improvement = (EO_gap_baseline - EO_gap_reweighed) / EO_gap_baseline × 100%
```

**Success Criterion**: EO_improvement ≥ 50%

### Step 6: Utility Preservation

**AUC Measurement**: 
```
AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt
```

**Utility Change**:
```
AUC_drop = AUC_baseline - AUC_reweighed
```

**Success Criterion**: AUC_drop ≤ 0.02 (2 percentage points)

---

## Mathematical Validation Steps

### 1. Weight Properties Validation

**Non-negativity**: ∀i: w(aᵢ, yᵢ) ≥ 0

**Positive sum**: Σᵢ w(aᵢ, yᵢ) > 0

**Group positivity**: For each existing (a,y) combination:
```
Σᵢ: aᵢ=a,yᵢ=y w(aᵢ, yᵢ) > 0
```

### 2. Independence Validation

For each (a, y) combination, check:
```
|P_w(A = a, Y = y) - P_w(A = a) × P_w(Y = y)| < ε
```

Where ε = 1e-3 (tolerance)

### 3. Correlation Reduction

**Original correlation**:
```
ρ_original = Σᵢ (aᵢ - μₐ)(yᵢ - μᵧ) / √[Σᵢ (aᵢ - μₐ)² × Σᵢ (yᵢ - μᵧ)²]
```

**Weighted correlation**:
```
ρ_weighted = Σᵢ wᵢ(aᵢ - μₐʷ)(yᵢ - μᵧʷ) / √[Σᵢ wᵢ(aᵢ - μₐʷ)² × Σᵢ wᵢ(yᵢ - μᵧʷ)²]
```

**Reduction percentage**:
```
Reduction% = (|ρ_original| - |ρ_weighted|) / |ρ_original| × 100%
```

---

## Complete Mathematical Workflow

```
1. Input: Biased dataset D = {(xᵢ, yᵢ, aᵢ)}ᵢ₌₁ⁿ

2. Compute empirical probabilities:
   P̂(A = a) = |{i: aᵢ = a}| / n
   P̂(Y = y) = |{i: yᵢ = y}| / n
   P̂(A = a, Y = y) = |{i: aᵢ = a ∧ yᵢ = y}| / n

3. Calculate reweighing weights:
   w(aᵢ, yᵢ) = P̂(A = aᵢ) × P̂(Y = yᵢ) / P̂(A = aᵢ, Y = yᵢ)

4. Validate weights:
   - Check non-negativity: w(aᵢ, yᵢ) ≥ 0
   - Check independence: |P_w(A,Y) - P_w(A)P_w(Y)| < ε

5. Train models:
   - Baseline: minimize Σᵢ ℓ(f(xᵢ), yᵢ)
   - Reweighed: minimize Σᵢ w(aᵢ, yᵢ) × ℓ(f(xᵢ), yᵢ)

6. Evaluate fairness:
   - Compute EO_gap = |TPR_group1 - TPR_group2|
   - Calculate improvement = (EO_baseline - EO_reweighed) / EO_baseline

7. Evaluate utility:
   - Compute AUC_drop = AUC_baseline - AUC_reweighed
   - Check preservation: AUC_drop ≤ 0.02

8. Success criteria:
   - Fairness: EO_improvement ≥ 50%
   - Utility: AUC_drop ≤ 2%
   - Overall: Both criteria met
```

---

## Key Mathematical Insights

1. **Reweighing Formula**: The ratio P(A)P(Y)/P(A,Y) mathematically enforces independence by making the weighted joint distribution equal the product of marginals.

2. **Weight Interpretation**: Weights > 1 upweight underrepresented combinations, weights < 1 downweight overrepresented combinations.

3. **Independence Guarantee**: By construction, reweighing achieves exact statistical independence in the weighted distribution.

4. **Fairness-Utility Trade-off**: The technique balances fairness improvement against utility preservation through the weighted loss function.

5. **Validation Framework**: Mathematical validation ensures the weights have correct statistical properties and achieve the desired independence.

This mathematical framework provides the complete theoretical foundation for understanding exactly what Task 5 accomplishes and how each step contributes to bias mitigation.