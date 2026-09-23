# Align under the pinned group, never a larger one

A note for the consistency side, from the regularisation side. It is one rule, and getting it
wrong does not produce an error — it produces a number that looks excellent.

## The rule

The indeterminacy class used to align two runs must be **the gauge group the objective actually
leaves free**, not a convenient superset. Aligning under a larger class absorbs, into the fitted
alignment, exactly the disagreement the measurement exists to detect.

## Why it bites here specifically

An objective that pins a frame does so by _shrinking_ the group. Whitening the state and matching
a product target on the residual gives `I = KKᵀ + D`, equivalently `K = ΔO`, and the four cases
come out:

| what is imposed            | gauge left free             | matching class |
| -------------------------- | --------------------------- | -------------- |
| whitening alone            | `O(L)`                      | `Orthogonal`   |
| product shape target alone | monomial `PΛ`               | **none**       |
| both                       | signed permutations (`B_L`) | `Permutation`  |
| neither                    | `GL(L)`                     | `Linear`       |

The whole point of the shape term is to move a run from row 1 to row 3 — from a continuous group
to a finite one. Scoring that run under `Linear` re-admits every element of `GL(L)` at alignment
time, which hands back for free precisely what the training paid to remove. The score then reports
that the two runs span a common subspace, which was never in question.

## The measurement, twice, from both sides

A deliberately scrambled frame — a genuine violation, not a gauge move:

| indeterminacy | score      | source                                   |
| ------------- | ---------- | ---------------------------------------- |
| `Linear`      | **1.0000** | blindness check on a frame-pinned latent |
| `Permutation` | **0.02**   | same pair, same data                     |

And on a pair differing by `A = diag(3, 1, 1, 1)`, driven through `ConsistencyEnsemble`:

| indeterminacy | score           |
| ------------- | --------------- |
| `Linear`      | **1.000000000** |
| `Orthogonal`  | **0.475266414** |

Same phenomenon at two different group sizes. `Linear` returns `1.0` in both, and in neither case
is that a false positive in the ordinary sense — `Linear` is answering its own question correctly.
It is the wrong question for a method whose claim is about the frame.

## Consequence for consensus averaging

The same rule governs averaging invariants across runs to cut residual variance. The average is
only meaningful as a quotient by the pinned group. Align under `Linear` first and the residual
variance that comes out is small for exactly the reason the alignment is uninformative: the
variation was absorbed, not measured. Frame-pinning is what makes a consensus latent worth
forming; scoring it under a superset gives the number away.

## Two practical riders

- **Floors differ by class, and one of them inverts the usual expectation.** `Permutation` chance
  sits at about `−1`, not near zero, so a `Permutation` score needs no chance caveat.
  `Linear` and `Affine` carry roughly `L/N` in-sample, so they must be scored **held out** or
  they flatter themselves in proportion to latent width.
- **There is a hole in the class list.** Nothing sits between `Permutation` and `Linear`. That is
  fine for a whitened latent, where the scale is already fixed and signed permutations are exact,
  and wrong for an unwhitened one, whose gauge is the monomial group `PΛ` — permutation with a
  free positive diagonal. `Permutation` is then too tight and `Linear` far too loose. A
  `Monomial` class (Procrustes on signs and order, then a per-coordinate scale) is the missing
  rung, and it is the one an anti-collapse-only objective actually needs.
