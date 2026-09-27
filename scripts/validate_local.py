"""Run the organizers' compliance and reproducibility checks against this checkout.

The official `verify_submission.py` clones a GitHub URL. This runs the same
checks locally so nothing is discovered only after pushing: it invokes the entry
point twice in a scratch directory, byte-compares both FASTA files, and applies
every sequence rule plus the reference-similarity gate.

    uv run python scripts/validate_local.py                # broad_spectrum
    uv run python scripts/validate_local.py --entry generate_mdr
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.compliance import ReferenceIndex, validate_library, validate_top  # noqa: E402
from amp_challenge_2027.constants import LIBRARY_SIZE, TOP_SIZE  # noqa: E402
from amp_challenge_2027.fasta import read_sequences  # noqa: E402


def run_entry_point(entry: str, out_dir: Path, extra: list[str]) -> None:
    cmd = ["uv", "run", "--no-sync", entry, "--out-dir", str(out_dir), *extra]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"entry point '{entry}' failed with code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="generate")
    ap.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    ap.add_argument("--top-k", type=int, default=TOP_SIZE)
    ap.add_argument("--skip-reproducibility", action="store_true")
    args = ap.parse_args()

    extra = ["--n-sequences", str(args.n_sequences), "--top-k", str(args.top_k), "--quiet"]
    scratch = Path(tempfile.mkdtemp(prefix="amp-validate-"))
    try:
        first, second = scratch / "run1", scratch / "run2"

        print("[1] First generation run")
        t0 = time.time()
        run_entry_point(args.entry, first, extra)
        print(f"    {time.time() - t0:.1f}s")

        print("[2] Loading reference database")
        reference = read_sequences(ROOT / "data" / "antibacterial.fasta")
        index = ReferenceIndex(reference)

        library = read_sequences(first / "library.fasta")
        top = read_sequences(first / "top.fasta")

        print(f"[3] Validating library ({len(library)} sequences)")
        validate_library(library, frozenset(reference), args.n_sequences)

        print(f"[4] Validating top list ({len(top)} sequences)")
        validate_top(top, frozenset(library), index, args.top_k)

        if args.skip_reproducibility:
            print("[5] Reproducibility check skipped")
        else:
            print("[5] Second generation run (reproducibility)")
            t0 = time.time()
            run_entry_point(args.entry, second, extra)
            print(f"    {time.time() - t0:.1f}s")
            for name in ("library.fasta", "top.fasta"):
                a = (first / name).read_bytes()
                b = (second / name).read_bytes()
                if a != b:
                    raise SystemExit(f"reproducibility check failed: {name} differs between runs")
                print(f"    {name}: byte-identical ({len(a)} bytes)")

        print("\nAll checks passed.")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
