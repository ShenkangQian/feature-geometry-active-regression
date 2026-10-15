# Implementation

## Shared representation and task adaptation

`GeometricFeatures` defines a feature vector with affine terms and tanh atoms. The default latent vector has two coordinates for position and two for log width. Shared parameters include base centers, base log widths, normalized latent loadings, readout prior mean, diagonal readout prior covariance, and diagonal latent prior scale.

The readout is Gaussian conditional on the geometry. `blr.py` computes its posterior and marginal likelihood with Cholesky solves. `inference.py` fits a diagonal Gaussian over the geometric coordinates using the collapsed likelihood and its KL term. Meta-training differentiates through functional inner updates and optimizes the context-conditioned query predictive density.

Training, validation, and test splits contain entire functions. Validation selects shared checkpoints and interval inflation factors. During testing, only the observed context updates the task posterior. Normalization is fitted on training tasks.

## Predictions and queries

A finite mixture of conditional Gaussian functions represents the task posterior. Every sampled function uses the same geometric state and readout sample across input locations. Total covariance includes expected conditional readout covariance and covariance between conditional means. For the mechanical simulator, real and imaginary readouts share geometric latents.

Moment acquisition uses the covariance of the moment-matched Gaussian. Fantasy acquisition conditions the finite Gaussian mixture on hypothetical observations and updates component weights. Observation noise enters the prospective measurement covariance; the optimization target is integrated latent-function error. Both rules mix in uniform exploration.

Candidate, integration, and evaluation grids are separate. Acquisition receives locations and queried observations. Independent evaluation targets remain inside the evaluator. Noise is keyed by task and location, so methods that request the same location receive the same realization.

## Comparators

| Setting | Definition |
|---|---|
| `fixed` | Shared feature map at `z = 0`, exact Bayesian readout |
| `full` | Joint position and width posterior adaptation |
| `position`, `width` | Adapt only the named latent coordinates |
| `point` | MAP geometry with conditional readout covariance |
| `linearized` | First-order geometry expansion at the context-conditioned readout mean |
| `linearized_prior` | Expansion at the prior readout mean, retained as a comparator diagnostic |
| `oracle_z` | Truth-fitted geometry with readout conditioned only on acquired observations |
| `full_cheap`, `full_rich` | Alternative inference step and particle budgets |
| `gp` | Two-scale Matérn-5/2 GP trained on the same training tasks |
| `gp_basis` | Matérn residual process plus the learned fixed feature basis |
| `random` | Full posterior with uniformly random queries |
| `<method>_random` | Named predictor with uniformly random queries |

`oracle_z` has evaluator access to the simulated truth and is a diagnostic reference. Its readout still has to be estimated from the context, so it is not a bound on predictive error. The fixed and linearized comparators are internal mechanism ablations; they do not reproduce the full training procedures of ALPaCA or LUMA.

## Metrics

The principal metric is the trapezoidal area under the latent-MSE curve divided by the observation-budget span. MSE is measured after training-only output normalization. `task_auc.csv` contains one row per task and method. `summary.csv` aggregates each budget, and `paired_comparisons.csv` resamples complete paired tasks.

Span residuals measure how much of the known synthetic function lies outside a chosen feature span. The held-out truth is used for these diagnostics only. A small span residual does not imply that the readout is identifiable from a few observations.

Coverage measures pointwise intervals for the noiseless function. Calibration multiplies standard deviations by a factor selected on validation tasks. Finite-mixture quantiles use deterministic Monte Carlo. Neither the finite candidate pool nor pointwise intervals certify that every narrow peak is observed.

The implementation handles one-dimensional inputs. The pulse functions and damped chains are simulated; the mechanical oracle is a numerical reference for an eight-mass linear system.

The [two diagnostic propositions](theory.md) give the translation remainder and fixed-span error bound used in the experiment design.
