"""FASTA reading and writing.

The reader is byte-for-byte compatible with the parser in the official
`scripts/verify_submission.py`, so what we validate locally is exactly what the
organizers will parse.
"""

from __future__ import annotations

from pathlib import Path


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Return (headers, sequences). Mirrors the official template's parser."""
    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    parts: list[str] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(parts))
            header, parts = line[1:], []
        else:
            parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(parts))

    return headers, sequences


def read_sequences(path: Path) -> list[str]:
    return read_fasta(path)[1]


def write_fasta(sequences: list[str], path: Path, prefix: str = "seq") -> None:
    """Write sequences with deterministic ``>prefix{i}`` headers.

    Uses '\\n' explicitly and writes in binary-safe text mode so the output is
    byte-identical across platforms — the organizers byte-compare two runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">{prefix}{i}\n{seq}\n")
