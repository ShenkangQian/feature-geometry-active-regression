# Research design: representation, inference, and measurement

A learned feature map can fail in two different ways. It may lack the shape needed for a new function, or it may express that shape but require more observations to identify the readout. Task adaptation changes the representation; the experiment asks whether that change also makes the function easier to estimate from a small context. The experiment is organized around separating these sources of error, then asking whether posterior uncertainty leads to better measurements.

## 1. Keep the adaptation small enough to examine

For `h` tanh atoms, the [model](../src/geometa/model.py) has `h + 2` readout features and four task-level geometric coordinates. It partitions the latent vector into position and width coordinates:

```text
z = [z_position, z_width]
c(z) = c_base + 0.25 A_position z_position
log ell(z) = log ell_base + 0.45 A_width z_width.
```

The loading columns are normalized by their root mean square with a small stabilizer. Log widths are clamped to `[-4.6, 0.4]` before exponentiation. These choices keep widths positive and bound the scales encountered during optimization. The center shifts remain linear in the task coordinates; nonlinearity enters through the resulting tanh features and width transformation.

Four coordinates can move several atoms together. Shared training learns those coordinated directions instead of allowing each new context to independently relocate all atoms. The feature vector also includes an intercept and a linear term, so smooth background variation need not be represented entirely by tanh combinations.

This structure provides interpretable interventions. Holding `z = 0` preserves the shared representation. Masking the width or position coordinates changes one kind of adaptation while keeping the checkpoint, readout update, and measurement interface fixed.

## 2. Integrate what can be solved exactly

For one output, write the context likelihood and priors as

```text
w ~ N(m0, S0)
z ~ N(0, Dz)
y_C | z, w ~ N(Phi_C(z) w, sigma² I).
```

`S0` and `Dz` are diagonal learned prior covariances. Marginalizing `w` gives

```text
y_C | z ~ N(Phi_C(z) m0,
             Phi_C(z) S0 Phi_C(z)ᵀ + sigma² I).
```

The readout need not be included in the variational family. [The inner optimizer](../src/geometa/inference.py) instead fits `q(z) = N(mu, diag(exp(2 log_std)))` by minimizing

```text
-E_q[log p(y_C | z)] + KL(q(z) || p(z)).
```

Reparameterized samples estimate the expectation, and the Gaussian KL is evaluated analytically. The code divides the whole objective by the number of observed scalar values. Scaling both terms together changes the update scale without changing their relative statistical weight. Fixed reparameterization samples within an adaptation call and bounded updates make this finite-step procedure reproducible.

After adaptation, each sampled `z` defines an exact conditional Gaussian readout posterior. The predictive approximation is a mixture of these conditional distributions. The Gaussian calculations are exact conditional on the selected geometry; the latent integral and its variational approximation remain numerical.

For the chain simulator, real and imaginary readouts are independent conditional on `z`, with a shared covariance structure and separate means. Marginalizing the shared geometry induces dependence between the two outputs. Dropping the conditional readout covariance would remove one source of uncertainty even if many latent particles were retained.

## 3. Train for the prediction made after adaptation

A [training episode](../src/geometa/training.py) samples a context and a separate query set from one training function. The context defines the inner update. The outer objective evaluates the conditional joint density of all query observations:

```text
L_outer = -log[(1 / M) sum_s p(y_Q | y_C, z_s)] / number_of_query_values,
z_s ~ q(z | context).
```

The distinction between a mixture of joint densities and a product of pointwise mixture densities is consequential. In the former, one latent geometry explains the entire query vector. In the latter, separate points could effectively choose different mixture components, giving a training objective that overlooks the function-level dependence needed for acquisition.

Functional inner updates retain the autograd graph during meta-training. The outer gradient therefore reaches both the shared representation and the priors through adaptation. The test `test_meta_gradients_reach_shared_features_and_priors` checks finite, nonzero gradients for all seven shared parameter groups. A validation set selects the shared checkpoint; test tasks can update their own latent posterior without changing that checkpoint or the training normalizer.

## 4. Make uncertainty answer a measurement question

Under squared error, the posterior mean is the Bayes predictor and its expected integrated loss is the integrated posterior variance. The acquisition objective therefore asks how much one observation is expected to reduce that variance.

For scalar outputs and a Gaussian posterior, the reduction at integration location `u` after querying `x` is

```text
C(u, x)² / (C(x, x) + sigma²).
```

Averaging over integration locations gives the score. [The implementation](../src/geometa/acquisition.py) uses the corresponding matrix expression for vector observations, accounting for the real and imaginary values measured together at one frequency.

A mixture introduces another effect: an observation changes the probabilities of its components. The fantasy rule samples prospective observations, conditions every Gaussian component, updates its weight by Bayes' rule, and evaluates the resulting mixture variance. Common random numbers across candidate locations reduce Monte Carlo ranking noise. This conditions the finite mixture; it does not refit the variational posterior for each fantasy.

Two tests establish useful reference cases. With one Gaussian component, fantasy acquisition agrees with analytic variance reduction. With two almost deterministic functions near `+2` and `-2`, one measurement identifies the component and removes almost all uncertainty elsewhere. A routine that updated only the readouts would miss that gain. Both cases are in [test_math.py](../tests/test_math.py).

## 5. Connect the theory to comparisons that can disagree

The [translation proposition](theory.md) bounds the first-order remainder of one tanh feature in terms of displacement divided by width. The fixed-span proposition uses orthogonal projection to lower-bound the error achievable by any readout on a chosen feature matrix. They motivate diagnostics, but neither establishes which fitted algorithm will win with a small context.

The recorded numerical span residual uses a small ridge stabilizer when solving the projection problem. It approximates the ideal least-squares residual in the proposition. Truth-fitted geometry uses bounded, multi-start optimization; it is a reference found by that search, not a certified global optimum. Even with a favorable geometry, its readout must still be inferred from the acquired observations.

The linearized comparator makes another distinction explicit. Its geometry Jacobian is evaluated at the context-conditioned readout mean. Expanding at a near-zero prior mean can suppress the geometry directions before the comparison begins. [The mechanism tests](../tests/test_mechanism.py) construct exactly that case and check that the conditional-mean expansion remains responsive. This is a context-dependent Gaussian surrogate, not exact inference in the nonlinear model.

The comparison matrix then separates geometry, approximation, and acquisition. Shared-checkpoint ablations isolate task adaptation. Richer inference changes steps and particles. A GP changes the inductive bias, and adding the fixed feature basis to that GP tests whether access to the learned basis explains its behavior. Random queries hold inference constant while changing measurement selection.

## 6. Read the evidence at the right level

The [sweep results](results.md) show why representation and estimation need separate diagnostics. At the narrowest width scale, the geometric posterior improves mean AUC from the fixed model's `0.408` to `0.316`; the GP obtains `0.259`. Across translation groups, the representation floor remains a small fraction of the fixed model's final-budget MSE at 10 observations; this ratio uses endpoint MSE rather than query-averaged AUC. Expressing the function and estimating it from three to ten observations are different requirements.

The analysis also separates individual tasks from group averages. Task-level correlations do not support a strong per-task prediction rule, while the exploratory group-level association suggests where representation pressure becomes more relevant. The [scorecard](../results/sweep-20260924/mechanism/scorecard.md) retains the rules and measured outcomes; group-level interpretation is not substituted for the original task-level tests.

Coverage is measured on independent targets for the noiseless function. The validation correction scales each component's standard deviation before forming mixture quantiles; it leaves the component means unchanged. Reporting this alongside the raw intervals helps distinguish interval adjustment from a change to the learned representation.

Every comparison uses whole functions as the evaluation unit. [Data tests](../tests/test_data_and_inference.py) check split isolation, deterministic task generation, and common query noise. The mechanical reference is checked by comparing modal evaluation with a direct complex solve. Together with source hashes and complete per-budget records, these checks make discrepancies traceable to model assumptions, numerical calculations, or experiment definitions.

For executable entry points, follow the [reproduction guide](reproduction.md). The [implementation reference](implementation.md) gives the complete method-label dictionary, and the [API example](../scripts/api_example.py) shows how to supply observed measurements and request a next location from a saved checkpoint.
