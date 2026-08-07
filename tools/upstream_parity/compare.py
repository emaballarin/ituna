"""Prove that the fork reports the same numbers as upstream, or say exactly where it does not.

The question this answers is NOT "is the fork correct?" but "did the fork change the measurement?".
Those have different answers and only the second one is at issue: a fix that repairs a crashing path
is free, because a path that always raised was never producing a number.

Method, borrowed from the sibling project's release discipline: materialise the reference tree at a
pinned ref, run one identical battery under each tree in a separate interpreter, and compare the
results as hex floats and sha256 digests -- never as printed decimals, which agree long before the
last bit does.

    python compare.py                      # against upstream main at 5aada31, offline
    python compare.py --ref 4858961        # isolate our own commits only
    python compare.py --mode upstream      # clone from dynamical-inference/ituna instead

WHICH REF. The diff has two strata: upstream's own unmerged feature branch (5aada31..4858961) and
our nine fix commits (4858961..HEAD). Default to 5aada31, which subsumes both and is the strongest
claim -- the feature branch is caching and backend routing, whose own contract is that results do
not change, so it ought to pass. If it fails, re-run against 4858961; agreement there says the
feature branch moved the number and our fixes did not.

WHAT A PASS MEANS. Only that the configurations in the battery agree. It is evidence, not proof, and
the battery is `probe.py` -- read it before quoting the verdict.
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
        # Offline and fast. `ref` is an ancestor of our HEAD, so its tree is already local.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="5aada31", help="reference commit (default: upstream main)")
    ap.add_argument("--mode", choices=["local", "upstream"], default="local")
    ap.add_argument("--keep", action="store_true", help="keep the temporary trees")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ituna-replicate-")
    try:
        ref_tree = materialise(args.ref, args.mode, os.path.join(tmp, "reference"))
        neutral = os.path.join(tmp, "cwd")
        os.makedirs(neutral, exist_ok=True)

        ref = run_probe(ref_tree, neutral)
        fork = run_probe(FORK, neutral)

        fp_ref, fp_fork = ref["fingerprint"], fork["fingerprint"]
        print("ARM FINGERPRINTS")
        for label, fp in (("reference", fp_ref), ("fork", fp_fork)):
            print(f"  {label:<10} {fp['ituna_file']}")
            print(f"  {'':<10} metrics.py sha256 {fp['metrics_sha256']}  numpy {fp['numpy']}")

        # THE GUARD. If both arms hashed the same metrics.py we compared a tree with itself, and
        # "they agree" would be true by construction. Agreement is the outcome we are hoping for,
        # which is exactly why this check has to be mechanical rather than a glance at the paths.
        if fp_ref["metrics_sha256"] == fp_fork["metrics_sha256"]:
            print("\nFATAL: both arms resolved to identical metrics.py. The comparison is vacuous.")
            return 3

        # Union the FAILURES too. Taking keys from the results alone means a run in which every
        # cell raised reports "0 cells, 0 differ" and exits 0 -- a clean-looking verdict from a
        # wholly broken run. That happened on the first execution of this harness.
        keys = sorted(set(ref["results"]) | set(fork["results"]) | set(ref["failures"]) | set(fork["failures"]))
        agree, differ, missing = [], [], []
        for key in keys:
            a, b = ref["results"].get(key), fork["results"].get(key)
            if a is None or b is None:
                missing.append(key)
            elif a == b:
                agree.append(key)
            else:
                differ.append((key, a, b))

        used = [k for k in agree if k.startswith("diag=False")]
        used_differ = [k for k, _, _ in differ if k.startswith("diag=False")]
        used_missing = [k for k in missing if k.startswith("diag=False")]
        print(f"\nBATTERY  {len(keys)} cells   agree {len(agree)}   differ {len(differ)}   incomparable {len(missing)}")
        print(f"THE PATH WE USE (include_diagonal=False): {len(used)} agree, {len(used_differ)} differ, {len(used_missing)} incomparable")

        if not used and not used_differ:
            print("\nFATAL: nothing on the used path was comparable. This is not a pass.")
            for key in missing[:5]:
                print(f"  {key}\n     reference {ref['failures'].get(key, 'absent')}\n     fork      {fork['failures'].get(key, 'absent')}")
            return 3

        for key, a, b in differ:
            print(f"\n  DIFFER  {key}")
            for field in sorted(set(a) | set(b)):
                if a.get(field) != b.get(field):
                    print(f"     {field}\n       reference {a.get(field)}\n       fork      {b.get(field)}")

        for key in missing:
            print(f"\n  INCOMPARABLE  {key}")
            print(f"     reference {ref['failures'].get(key, 'absent')}")
            print(f"     fork      {fork['failures'].get(key, 'absent')}")

        print("\nEXPECTED DIFFERENCES, if they appear above:")
        print("  * source_id_result -- upstream raises AttributeError on every input; the fork")
        print("    returns a value. A repaired crash cannot have moved a number.")
        print("  * diag=True cells -- the fork's _score ignores the diagonal unconditionally,")
        print("    matching its own docstring. Upstream inflates to (1-f)s+f. We never pass True.")

        return 0 if not (used_differ or used_missing) else 1
    finally:
        if args.keep:
            print(f"\ntrees kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
