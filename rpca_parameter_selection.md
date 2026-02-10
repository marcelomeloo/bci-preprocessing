# RPCA Parameter Selection Methodology

## Overview
Optimal lambda (λ) and mu (μ) parameters for RPCA were selected using grid search optimization on BCI spectrogram data.

## Grid Search Process

### Parameter Space
- **Lambda (λ)**: Predefined values from `lamb_values_spectogram`
- **Mu (μ)**: Calculated as `μ = λ / denominator` using `mu_denominators`

### Optimization Loop
For each preprocessing method (RPCA_L, RPCA_S, CAR_RPCA_L, CAR_RPCA_S):

1. **Initialize**: `λ_best = λ_zero/2`, `μ_best = λ_best/2`, `accuracy_max = 0`
2. **Grid Search**: Test all (λ, μ) combinations
3. **Pipeline**: 
   - Apply RPCA filtering with current parameters
   - Extract spectrogram features 
   - Train linear SVM classifier
   - Evaluate on validation set (20% holdout)
4. **Update**: If `accuracy_current > accuracy_max`, save current (λ, μ)

## Selection Criterion
**SVM classification accuracy** on validation data using stratified train/test split (80/20) with fixed random seed (42).

## Output
Best parameters maximizing discriminative power between different evoked frequencies for accurate BCI classification.






