"""Run a fixed battery over one ituna tree and print the results as JSON.

Invoked once per arm by `compare.py`; never useful on its own. The tree to test is argv[1], and it
is placed at the FRONT of sys.path and then verified -- an installed `ituna` in the ambient venv
would otherwise shadow it silently and both arms would resolve to the same code, which is the exact
way a differential check passes tautologically.

Every cell is wrapped, so a call that raises in one tree and succeeds in the other is recorded as a
compared value rather than killing the run. That matters here: one of the fork's fixes repairs a
public call that always raised upstream, and the harness should surface that as a difference it can
explain, not as a crash.

The import of `ituna` lives inside `load()` rather than at module scope, because it can only happen
after sys.path is pointed at the tree under test -- and a module-level import that late is exactly
what a linter is right to complain about.
"""

import hashlib
import json
import os
import sys

import numpy as np

N_SAMPLES, N_FEATURES = 256, 6
CLASSES = ["Identity", "Permutation", "Linear", "Affine"]
REGIMES = ["identical", "signed_permutation", "rotation", "independent"]
K_VALUES = [2, 3, 5]


def load(tree):
    """Import ituna from `tree`, then prove that is where it came from."""
    sys.path.insert(0, tree)
    import ituna
    from ituna import metrics

    resolved = os.path.abspath(os.path.dirname(ituna.__file__))
    if os.path.commonpath([resolved, tree]) != tree:
        print(json.dumps({"fatal": f"ituna resolved to {resolved}, not under {tree}"}))
        sys.exit(2)
    return ituna, metrics


def digest(a):
    """sha256 of an array's exact bytes, shape and dtype included so a reshape cannot alias."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:32]


def meta(obj, name):
    """Read PairwiseConsistencyArray metadata. Indirect because the abstract base stubs the
    transform return type, so a direct attribute access is unresolvable to a checker."""
    return getattr(obj, name)


def make_spaces(regime, k):
    """K embeddings related to each other in a way the indeterminacy classes can be told apart on.

    `rotation` is the interesting one: consistent under Linear/Affine and NOT under Permutation,
    which is the asymmetry the whole exercise is about.
    """
    base = np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
    out = []
    for i in range(k):
        rng = np.random.default_rng(1000 * (REGIMES.index(regime) + 1) + i)
        if regime == "identical":
            out.append(base.copy())
        elif regime == "signed_permutation":
            perm = rng.permutation(N_FEATURES)
            signs = rng.choice([-1.0, 1.0], size=N_FEATURES)
            out.append(base[:, perm] * signs)
        elif regime == "rotation":
            q, _ = np.linalg.qr(rng.standard_normal((N_FEATURES, N_FEATURES)))
            out.append(base @ q)
        elif regime == "independent":
            out.append(rng.standard_normal((N_SAMPLES, N_FEATURES)))
    return out


def cell(metrics, cls_name, regime, k, symmetric, include_diagonal, probe_source_id):
    """One configuration, reduced to values that compare exactly across trees."""
    spaces = make_spaces(regime, k)
    pc = metrics.PairwiseConsistency(
        indeterminacy=getattr(metrics, cls_name)(),
        symmetric=symmetric,
        include_diagonal=include_diagonal,
        random_state=0,
    )
    pc.fit(spaces)
    out = {}

    score = pc.score(spaces)
    # Hex, because two floats can print identically at repr precision and differ in the last bit.
    out["score_hex"] = float(score).hex()
    out["score_isnan"] = bool(np.isnan(score))

    arr = pc.transform(spaces)
    out["reference_id"] = int(meta(arr, "reference_id"))
    out["mean_embedding_sha"] = digest(np.asarray(arr))
    # `scores` is a tuple (pair_indices, per_pair_scores), NOT a K x K matrix. Digesting both pins
    # which pairs were fitted as well as what they scored, which is strictly more than the matrix.
    pair_indices, pair_scores = meta(arr, "scores")
    out["pair_indices_sha"] = digest(np.asarray(pair_indices, dtype=np.int64))
    out["pair_scores_sha"] = digest(np.asarray(pair_scores, dtype=np.float64))

    if probe_source_id:
        # Upstream raises AttributeError here for every input; the fork returns a value. A repaired
        # crash cannot have moved a number, so this cell is expected to differ and is reported apart.
        try:
            single = pc.transform([spaces[0]], source_id=0)
            out["source_id_result"] = digest(np.asarray(single))
        except Exception as exc:  # noqa: BLE001 -- the exception IS the observation
            out["source_id_result"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    tree = os.path.abspath(sys.argv[1])
    ituna, metrics = load(tree)

    results, failures = {}, {}
    for include_diagonal in (False, True):
        for cls_name in CLASSES:
            for regime in REGIMES:
                for k in K_VALUES:
                    for symmetric in (False, True):
                        key = f"diag={include_diagonal}|{cls_name}|{regime}|K={k}|sym={symmetric}"
                        try:
                            results[key] = cell(
                                metrics,
                                cls_name,
                                regime,
                                k,
                                symmetric,
                                include_diagonal,
                                probe_source_id=(k == 3 and not symmetric and not include_diagonal),
                            )
                        except Exception as exc:  # noqa: BLE001
                            failures[key] = f"{type(exc).__name__}: {exc}"

    with open(os.path.join(tree, "ituna", "metrics.py"), "rb") as fh:
        metrics_sha = hashlib.sha256(fh.read()).hexdigest()[:32]

    print(
        json.dumps(
            {
                "fingerprint": {
                    "tree": tree,
                    "ituna_file": ituna.__file__,
                    "metrics_sha256": metrics_sha,
                    "numpy": np.__version__,
                    "classes_present": [c for c in CLASSES if hasattr(metrics, c)],
                },
                "results": results,
                "failures": failures,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
