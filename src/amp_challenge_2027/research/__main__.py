"""python -m amp_challenge_2027.research {prepare,embed,develop,evaluate-test}"""

import argparse
import json
from pathlib import Path

from .data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze data provenance and family partitions")
    prep.add_argument("--source", type=Path, default=Path("data/grampa.csv"))
    prep.add_argument("--family-threshold", type=float, default=0.7)
    embed = commands.add_parser("embed", help="Cache revision-pinned ESM embeddings in resumable shards")
    from .embeddings import DEFAULT_MODEL, DEFAULT_REVISION
    embed.add_argument("--model", default=DEFAULT_MODEL)
    embed.add_argument("--revision", default=DEFAULT_REVISION)
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--local-only", action="store_true")
    for name in ("develop", "evaluate-test"):
        command = commands.add_parser(name)
        command.add_argument("--species", default="E. coli")
        command.add_argument("--embeddings", type=Path)
        command.add_argument("--top-k", type=int, default=50)
        command.add_argument("--ensemble-members", type=int, default=5)
        command.add_argument("--mlp-iterations", type=int, default=1200)
    for command in commands.choices.values():
        command.add_argument("--work", type=Path, default=Path("research_runs/challenger"))
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.work, args.family_threshold, args.seed)
        print(json.dumps(result["audit"], indent=2))
    elif args.command == "embed":
        from .embeddings import embed as extract
        extract(args.work, args.model, args.revision, args.batch_size, args.device, args.local_only)
    else:
        from .experiment import run
        report = run(args.work, args.species, args.embeddings, args.top_k,
                     args.ensemble_members, args.mlp_iterations, args.seed,
                     final=args.command == "evaluate-test")
        print(json.dumps({"stage": report["stage"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
