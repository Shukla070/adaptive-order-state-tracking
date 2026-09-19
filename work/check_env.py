"""
Does this environment match the one the results were produced in?

Compares the live interpreter against requirements.lock.txt and against the
versions recorded in ENVIRONMENT.md, and reports every difference. Run it after
rebuilding an environment, before spending GPU hours in it.

A rebuilt environment that silently drifted is the expensive failure: the code
imports, the tests pass, the run completes, and the numbers are not comparable
with the ones already recorded.

    python work/check_env.py
    python work/check_env.py --strict     # any drift at all is a failure

Exit code 0 when the packages that matter agree, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Packages whose version changes the numbers, not just the packaging. A
# mismatch in any of these is a hard failure; anything else is reported as
# drift and tolerated unless --strict.
CRITICAL = {
    "torch",           # kernels, autograd, dtype behaviour
    "triton",          # the chunked kernels are Triton
    "numpy",
    "polars",          # the data loader
    "transformers",    # the config/model plumbing fla builds on
    "einops",
}

REPO = Path(__file__).resolve().parent.parent


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        name, _, ver = line.partition("==")
        out[normalise(name)] = ver.strip()
    return out


def installed_versions() -> dict[str, str]:
    from importlib.metadata import distributions
    out: dict[str, str] = {}
    for d in distributions():
        n = d.metadata["Name"]
        if n:
            out[normalise(n)] = d.version
    return out


def recorded_env(path: Path) -> dict[str, str]:
    """Pull the version table out of ENVIRONMENT.md."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", line)
        if m and m.group(1) not in ("", "---"):
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat any version difference as a failure")
    args = ap.parse_args()

    lock_path = REPO / "requirements.lock.txt"
    env_path = REPO / "ENVIRONMENT.md"

    print("=" * 74)
    print("ENVIRONMENT — does this match the one the results came from?")
    print("=" * 74)

    if not lock_path.exists():
        print(f"\n  no {lock_path.name} found.")
        print("  Capture one from a working environment first:")
        print("      bash work/capture_env.sh")
        return 1

    lock = read_lock(lock_path)
    live = installed_versions()
    rec = recorded_env(env_path)
    print(f"  lock file: {len(lock)} packages pinned\n")

    missing, differing, drift = [], [], []
    for name, want in sorted(lock.items()):
        got = live.get(name)
        if got is None:
            (missing if name in CRITICAL else drift).append((name, want, "absent"))
        elif got != want:
            (differing if name in CRITICAL else drift).append((name, want, got))

    # ---- the packages that change the numbers -------------------------------
    print("  packages that affect results")
    ok = True
    for name in sorted(CRITICAL):
        want, got = lock.get(name), live.get(name)
        if want is None:
            print(f"    --   {name:<16} not in the lock file")
            continue
        if got is None:
            print(f"    FAIL {name:<16} want {want}, NOT INSTALLED")
            ok = False
        elif got != want:
            print(f"    FAIL {name:<16} want {want}, got {got}")
            ok = False
        else:
            print(f"    ok   {name:<16} {got}")

    # ---- torch's own build details ------------------------------------------
    print("\n  runtime")
    try:
        import torch
        cuda_build = torch.version.cuda
        print(f"    torch {torch.__version__}  (CUDA build {cuda_build})")
        want_torch = rec.get("PyTorch")
        want_cuda = rec.get("PyTorch CUDA runtime")
        if want_torch and want_torch != torch.__version__:
            print(f"    FAIL recorded torch was {want_torch}")
            ok = False
        if want_cuda and want_cuda not in ("?", "None") and want_cuda != str(cuda_build):
            print(f"    FAIL recorded CUDA build was {want_cuda}")
            ok = False
        if torch.cuda.is_available():
            print(f"    GPU visible: {torch.cuda.get_device_name(0)}")
        else:
            print("    GPU NOT visible — CPU-only checks will run, "
                  "training will not")
    except Exception as e:                                   # noqa: BLE001
        print(f"    FAIL could not import torch: {e}")
        ok = False

    # ---- the vendored editable install --------------------------------------
    print("\n  vendored flash-linear-attention")
    try:
        import fla
        loc = Path(getattr(fla, "__file__", "") or "").resolve()
        inside = REPO in loc.parents
        print(f"    imported from {loc}")
        if inside:
            print("    ok   resolves to this repository's patched copy")
        else:
            print("    FAIL resolves OUTSIDE this repository — the patched copy")
            print("         is not what is being imported. Reinstall with:")
            print("         pip install --no-build-isolation -e ./flash-linear-attention")
            ok = False
    except Exception as e:                                   # noqa: BLE001
        print(f"    FAIL could not import fla: {e}")
        ok = False

    # ---- the compiled CUDA extension -----------------------------------------
    # Not optional: ShortConvolution raises at construction without it, and the
    # layer builds one by default. An environment that imports everything else
    # cleanly still cannot build a model if this is missing.
    print("\n  compiled extension")
    try:
        import causal_conv1d
        print(f"    ok   causal_conv1d {getattr(causal_conv1d, '__version__', '?')}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    FAIL causal_conv1d does not import: {e}")
        print("         ShortConvolution will raise at model construction.")
        print("         pip install --no-build-isolation causal-conv1d==1.7.0")
        ok = False

    # ---- the two --no-deps git packages --------------------------------------
    # Installed without their declared dependencies, so confirm they import
    # rather than assuming the install succeeded.
    print("\n  git dependencies (installed with --no-deps)")
    for mod in ("abstract_algebra", "sfirah"):
        try:
            __import__(mod)
            print(f"    ok   {mod} imports")
        except Exception as e:                               # noqa: BLE001
            print(f"    FAIL {mod} does not import: {e}")
            print("         reinstall with:")
            print("           pip install --no-deps git+https://github.com/jopetty/"
                  f"{mod}")
            ok = False

    # ---- everything else -----------------------------------------------------
    if missing or differing or drift:
        print("\n  other differences")
        for name, want, got in missing + differing + drift:
            print(f"    {name:<24} lock {want:<18} live {got}")
    else:
        print("\n  every pinned package matches exactly")

    if args.strict and drift:
        print("\n  --strict: the differences above are treated as failures")
        ok = False

    print("\n" + "=" * 74)
    if ok:
        print("MATCH — results produced here are comparable with those in results/.")
        print("Next: python work/check_gate_precision.py")
    else:
        print("MISMATCH — fix the FAIL lines before running experiments here.")
        print("Numbers from a drifted environment cannot be compared with the")
        print("ones already recorded, and the difference will not be obvious.")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
