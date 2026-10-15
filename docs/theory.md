# Two diagnostics for feature adaptation

The propositions referenced in `PREDICTIONS.md` describe a translated feature and a fixed feature span. They motivate the controlled sweeps; the experiment tests whether their quantities explain the fitted model's behavior.

## Proposition 1: translation relative to width

Let `h(x) = tanh((x - c) / ell)`, with positive width `ell`. Moving the center by `delta` gives the first-order approximation

```text
T_delta(x) = h(x) - (delta / ell) sech²((x - c) / ell).
```

The uniform Taylor remainder obeys

```text
|h_(c+delta,ell)(x) - T_delta(x)| <= 2 / (3 sqrt(3)) * (delta / ell)².
```

At the old center, the error divided by the magnitude of the first-order change is `1 - tanh(r)/r`, where `r = |delta|/ell > 0`. This ratio approaches one as `r` increases.

To obtain the bound, differentiate `g(u) = tanh(u)` twice. The absolute second derivative is `2v(1-v²)` for `v = |tanh(u)|`, whose maximum is `4/(3 sqrt(3))`. Taylor's theorem contributes a factor of one half. At `x = c`, the exact translated feature is `-tanh(delta/ell)`, while the approximation is `-delta/ell`.

This calculation concerns one feature. Other features and readout changes may compensate for its local approximation error, which is why the multi-feature comparison is empirical.

## Proposition 2: error outside a fixed span

Let `F` be the feature matrix on a fixed evaluation grid, `D` a positive diagonal integration-weight matrix, and `f` the function values. Write `A = sqrt(D) F`, `b = sqrt(D) f`, and let `P_A` project orthogonally onto the column space of `A`. Every readout vector `u` satisfies

```text
||sqrt(D) (f - F u)||² >= ||(I - P_A) b||².
```

The decomposition `b - A u = (I - P_A)b + (P_A b - A u)` separates orthogonal components. Pythagoras gives the bound, and a least-squares readout attains it without requiring full column rank. Taking expectations gives the same lower bound for a random readout supported on that fixed span.

The recorded `span_residual_z0`, `span_residual_zhat`, and `span_residual_zstar` evaluate this diagnostic at fixed, inferred, and truth-fitted geometry. The bound distinguishes representation error from estimation error. It does not say that a small context will identify the least-squares readout or find a better span.
