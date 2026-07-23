# 0001 — Fused sigmoid+BCE gradient for the output layer

## Status

Accepted

## Context

`backward_network` needs the gradient of Binary Cross-Entropy loss with
respect to the output layer's pre-activation `Z2`. There are two ways to
get there:

1. **Modular**: compute `dL/dA2` (the BCE derivative w.r.t. the
   sigmoid output), then chain it through `sigmoid_derivative(Z2)` to
   get `dL/dZ2`.
2. **Fused**: use the algebraic simplification `dZ2 = A2 - y`, which
   falls out when you multiply the BCE derivative by the sigmoid
   derivative and the terms cancel.

`binary_cross_entropy` (`loss.c`) clamps its prediction input to
`[EPSILON, 1-EPSILON]` specifically to keep `log(p)` finite. The
modular derivative of BCE w.r.t. `A2` is `-(y/A2) + (1-y)/(1-A2)` —
a `1/A2`-shaped division that reintroduces the exact `p → 0` or
`p → 1` blow-up the clamp exists to prevent. Chaining that through
`sigmoid_derivative` doesn't fix it: as `A2` approaches 0 or 1,
`sigmoid_derivative(Z2) = A2*(1-A2)` shrinks at the same rate the BCE
derivative grows, so the two divisions are precariously close to
cancelling analytically but not numerically once float rounding is in
the mix.

## Decision

`backward_network` uses the fused shortcut `dZ2 = A2 - y` for the
output layer's gradient, not a generically chained
`bce_derivative(A2) * sigmoid_derivative(Z2)`. This is the same
gradient `logistic_regression.c` already computes
(`compute_logistic_weight_gradient`), so the two models now share the
same numerically-stable pattern.

The hidden layer has no such shortcut — ReLU's derivative doesn't
cancel with anything upstream — so it chains through the real
`relu_derivative(Z1)`.

## Consequences

- The output layer's backward pass is a closed-form subtraction, not a
  composition through `sigmoid_derivative`, so `sigmoid_derivative` is
  exercised directly only by its own unit test, not by
  `backward_network`.
- `backward_network` is only correct for a sigmoid-output +
  BCE-loss network. Swapping the output activation or loss function
  would require deriving a new fused gradient, or falling back to the
  modular path for that combination.
- Gradient checking (`tests/test_gradients.c`) validates the fused
  shortcut against numerical differentiation of the actual BCE loss,
  so the algebraic simplification is checked end-to-end rather than
  trusted by derivation alone.
