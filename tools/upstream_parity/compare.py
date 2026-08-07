"""Check that the fork reports the same numbers as upstream, or say exactly where it does not.

The question is not "is the fork correct?" but "does the fork change what a correct run reports?".
Only the second one matters for anyone adopting it: a fix that repairs a crashing path is free,
because a path that always raised was never producing a number to move.

Method: materialise the reference tree at a pinned ref, run one identical battery under each tree in
a separate interpreter, and compare the results as hex floats and sha256 digests -- never as printed
decimals, which agree long before the last bit does.

    python compare.py                      # against upstream main at 5aada31, offline
    python compare.py --ref 4858961        # compare against the feature-branch merge instead
    python compare.py --mode upstream      # clone from upstream rather than using local history
    python compare.py --verbose            # list every explained difference, not just the tally

VERDICT. Every difference must be accounted for by a named mechanism, and each mechanism is
*verified* rather than asserted:

  * `source_id` -- upstream's `_get_indeterminancy` reads an attribute `_fit` never sets, so the
    call raises for every input. Counted as explained only when the reference side really is that
    AttributeError.
  * the diagonal, on `include_diagonal=True` -- upstream's `_score` averages self-alignments, which
    score 1.0 by construction. Counted as explained only when the observed pair satisfies
    `upstream == (1-f) * fork + f` exactly, with `f = 1/K` or `2/(K+1)`.

Anything else is UNEXPLAINED and fails. So does an incomparable cell, and so does a run in which
nothing was comparable at all.

WHICH REF. The history has two strata: upstream's own unmerged feature branch (5aada31..4858961) and
this fork's fixes (4858961..HEAD). The default subsumes both. If it ever fails, re-run against
4858961 to see which stratum moved the number.

WHAT A PASS MEANS. That the configurations in the battery agree. It is evidence, not proof, and the
battery is `probe.py` -- read it before quoting the verdict.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.abspath(os.path.join(HERE, "..", ".."))
UPSTREAM_URL = "https://github.com/dynamical-inference/ituna"


def materialise(ref, mode, dest):
    """Write the reference tree at `ref` into `dest`."""
    os.makedirs(dest, exist_ok=True)
    if mode == "upstream":
        # Strictly stronger: the bytes come from upstream's server, not from our object store.
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, dest], check=True)
        subprocess.run(["git", "-C", dest, "checkout", "--quiet", ref], check=True)
    else:
        # Offline and fast. `ref` is an ancestor of HEAD, so its tree is already local.
        archive = subprocess.run(["git", "-C", FORK, "archive", ref], check=True, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", dest], input=archive.stdout, check=True)
    return dest


def run_probe(tree, workdir):
    """Run the battery under one tree, in its own interpreter, from a neutral cwd."""
    env = dict(os.environ)
    # Neutralise anything that could inject a different ituna ahead of the tree under test.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "probe.py"), tree],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"probe failed for {tree} (rc={proc.returncode})\n{proc.stderr}")
    return json.loads(proc.stdout)


def parse_key(key):
    """`diag=False|Affine|identical|K=3|sym=False` -> the parts the mechanisms need."""
    parts = key.split("|")
    return {
        "diag": parts[0].split("=")[1] == "True",
        "K": int(parts[3].split("=")[1]),
        "sym": parts[4].split("=")[1] == "True",
    }


def diagonal_share(k, symmetric):
    """Fraction of the fitted pairs that are self-alignments: K of K(K+1)/2, or K of K**2."""
    return 2.0 / (k + 1) if symmetric else 1.0 / k


def classify(key, a, b):
    """Return (verdict, notes). Every difference is explained by a verified mechanism, or it is not.

    verdict is "identical", "explained" or "UNEXPLAINED".
    """
    meta = parse_key(key)
    diffs = [f for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
    if not diffs:
        return "identical", []

    notes, unexplained = [], []
    for field in diffs:
        if field == "source_id_result":
            # Explained only if upstream really cannot execute the call. If it ever starts
            # returning a value, this stops being a repaired crash and becomes a real difference.
            if str(a.get(field, "")).startswith("AttributeError"):
                notes.append("source_id: upstream raises for every input, the fork returns a value")
            else:
                unexplained.append(f"{field} (upstream did not raise -- not a repaired crash)")
        elif field == "score_hex" and meta["diag"]:
            frac = diagonal_share(meta["K"], meta["sym"])
            fork_score, ref_score = float.fromhex(b[field]), float.fromhex(a[field])
            predicted = (1.0 - frac) * fork_score + frac
            if abs(predicted - ref_score) <= 1e-12 * max(1.0, abs(ref_score)):
                notes.append(f"diagonal identity holds exactly (f={frac:.4f})")
            else:
                unexplained.append(f"{field} (diagonal identity does NOT hold)")
        else:
            unexplained.append(field)

    if unexplained:
        return "UNEXPLAINED", unexplained
    return "explained", notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="5aada31", help="reference commit (default: upstream main)")
    ap.add_argument("--mode", choices=["local", "upstream"], default="local")
    ap.add_argument("--keep", action="store_true", help="keep the temporary trees")
    ap.add_argument("--verbose", action="store_true", help="list every explained difference")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ituna-replicate-")
    try:
        ref_tree = materialise(args.ref, args.mode, os.path.join(tmp, "reference"))
        neutral = os.path.join(tmp, "cwd")
        os.makedirs(neutral, exist_ok=True)

        ref = run_probe(ref_tree, neutral)
        fork = run_probe(FORK, neutral)

        fp_ref, fp_fork = ref["fingerprint"], fork["fingerprint"]
        print(f"ARM FINGERPRINTS   (reference ref: {args.ref}, mode: {args.mode})")
        for label, fp in (("reference", fp_ref), ("fork", fp_fork)):
            print(f"  {label:<10} {fp['ituna_file']}")
            print(f"  {'':<10} metrics.py sha256 {fp['metrics_sha256']}  numpy {fp['numpy']}")

        # GUARD 1. If both arms hashed the same metrics.py we compared a tree with itself, and
        # "they agree" would be true by construction. Agreement is the outcome we are hoping for,
        # which is exactly why this has to be mechanical rather than a glance at the paths.
        if fp_ref["metrics_sha256"] == fp_fork["metrics_sha256"]:
            print("\nFATAL: both arms resolved to identical metrics.py. The comparison is vacuous.")
            return 3

        # GUARD 2. Union the failures into the key set. Taking keys from the results alone means a
        # run in which every cell raised reports "0 cells, 0 differ" and exits 0 -- a clean-looking
        # verdict from a wholly broken run.
        keys = sorted(set(ref["results"]) | set(fork["results"]) | set(ref["failures"]) | set(fork["failures"]))

        identical, explained, unexplained, missing = [], [], [], []
        for key in keys:
            a, b = ref["results"].get(key), fork["results"].get(key)
            if a is None or b is None:
                missing.append(key)
                continue
            verdict, detail = classify(key, a, b)
            {"identical": identical, "explained": explained, "UNEXPLAINED": unexplained}[verdict].append(key if verdict == "identical" else (key, detail))

        print(
            f"\nBATTERY  {len(keys)} cells   identical {len(identical)}   "
            f"explained {len(explained)}   UNEXPLAINED {len(unexplained)}   incomparable {len(missing)}"
        )

        mechanisms = {}
        for _, detail in explained:
            for note in detail:
                mechanisms[note.split(" (")[0].split(":")[0]] = mechanisms.get(note.split(" (")[0].split(":")[0], 0) + 1
        for name, count in sorted(mechanisms.items()):
            print(f"  explained by {name}: {count}")

        default_path = [k for k in keys if k.startswith("diag=False")]
        default_bad = [k for k, _ in unexplained if k.startswith("diag=False")] + [k for k in missing if k.startswith("diag=False")]
        print(f"DEFAULT PATH (include_diagonal=False): {len(default_path) - len(default_bad)}/{len(default_path)} accounted for")

        if not identical and not explained:
            print("\nFATAL: nothing was comparable. This is not a pass.")
            for key in missing[:5]:
                print(f"  {key}\n     reference {ref['failures'].get(key, 'absent')}\n     fork      {fork['failures'].get(key, 'absent')}")
            return 3

        if args.verbose:
            for key, detail in explained:
                print(f"\n  EXPLAINED  {key}")
                for note in detail:
                    print(f"     {note}")

        for key, detail in unexplained:
            a, b = ref["results"][key], fork["results"][key]
            print(f"\n  UNEXPLAINED  {key}")
            for field in detail:
                name = field.split(" (")[0]
                print(f"     {field}\n       reference {a.get(name)}\n       fork      {b.get(name)}")

        for key in missing:
            print(f"\n  INCOMPARABLE  {key}")
            print(f"     reference {ref['failures'].get(key, 'absent')}")
            print(f"     fork      {fork['failures'].get(key, 'absent')}")

        if unexplained or missing:
            print("\nFAIL: at least one difference is not accounted for by a verified mechanism.")
            return 1
        print("\nPASS: every difference is accounted for by a verified mechanism.")
        return 0
    finally:
        if args.keep:
            print(f"\ntrees kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
