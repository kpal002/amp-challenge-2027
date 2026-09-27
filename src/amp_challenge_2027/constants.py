"""Competition constants and the token vocabulary.

The sequence rules mirror `scripts/verify_submission.py` from the official
template exactly. They are duplicated here (rather than imported) so that the
generation path can enforce them itself and never emit an invalid library.
"""

from __future__ import annotations

# The 20 standard proteinogenic amino acids, in a fixed canonical order.
# The order defines token ids, so it must never change once weights are trained.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_SET = frozenset(AMINO_ACIDS)

MIN_LENGTH = 8
MAX_LENGTH = 50

LIBRARY_SIZE = 50_000
TOP_SIZE = 100

# Top-100 candidates must not exceed this similarity to any known reference AMP.
# The template implements the rule as Levenshtein.ratio; the competition proposal
# describes it as MMseqs2 alignment identity. These are different measures, so we
# apply a margin below the stated 0.80 to stay valid under either reading.
MAX_REFERENCE_SIMILARITY = 0.80
SIMILARITY_MARGIN = 0.05
TOP_SIMILARITY_CEILING = MAX_REFERENCE_SIMILARITY - SIMILARITY_MARGIN

# --- Token vocabulary -------------------------------------------------------
# Layout: [PAD, BOS, EOS, <control tokens...>, <amino acids...>]
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

# Category control tokens. One model serves every competition category; the
# control token selects the conditional distribution at sampling time.
CATEGORIES = (
    "broad_spectrum",
    "gram_pos",
    "gram_neg",
    "mdr",
    "therapeutic",
)
CONTROL_TOKENS = tuple(f"<{c}>" for c in CATEGORIES)

VOCAB: tuple[str, ...] = (PAD, BOS, EOS) + CONTROL_TOKENS + tuple(AMINO_ACIDS)
STOI = {tok: i for i, tok in enumerate(VOCAB)}
ITOS = {i: tok for tok, i in STOI.items()}

PAD_ID = STOI[PAD]
BOS_ID = STOI[BOS]
EOS_ID = STOI[EOS]
AA_IDS = tuple(STOI[a] for a in AMINO_ACIDS)
FIRST_AA_ID = AA_IDS[0]

VOCAB_SIZE = len(VOCAB)

# BOS + control + up to MAX_LENGTH residues + EOS
BLOCK_SIZE = MAX_LENGTH + 3


def control_id(category: str) -> int:
    """Token id of the control token for `category`."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    return STOI[f"<{category}>"]


def encode(sequence: str) -> list[int]:
    """Amino-acid string -> token ids (no BOS/EOS/control)."""
    return [STOI[c] for c in sequence]


def decode(ids: list[int] | tuple[int, ...]) -> str:
    """Token ids -> amino-acid string, stopping at EOS and skipping non-residues."""
    out: list[str] = []
    for i in ids:
        if i == EOS_ID:
            break
        tok = ITOS.get(int(i), "")
        if len(tok) == 1:
            out.append(tok)
    return "".join(out)
