#!/usr/bin/env python3
"""
NEXUS — Main CLI Entrypoint

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./sample_candidates.json --out ./test_output.csv --eval
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose

Reproduce command (for submission_metadata.yaml):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS — Intelligent Candidate Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidates", "-c",
        type=Path,
        required=True,
        help="Path to candidates.jsonl or sample_candidates.json",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        default=False,
        help="Run local NDCG evaluation after ranking (for development only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print progress (default: True)",
    )
    parser.add_argument(
        "--validate", "-V",
        action="store_true",
        default=True,
        help="Run validate_submission.py on output after writing (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate input file
    if not args.candidates.exists():
        print(f"[ERROR] Candidates file not found: {args.candidates}", file=sys.stderr)
        return 1

    # Run pipeline
    from nexus.pipeline import run_pipeline
    results = run_pipeline(
        input_path=args.candidates,
        output_path=args.out,
        verbose=args.verbose,
    )

    # Optional: local NDCG evaluation
    if args.eval:
        print("\nRunning local NDCG evaluation...")
        from nexus.evaluator import evaluate_ranking
        top100_candidates = [c for _, c, _ in results[:100]]
        metrics = evaluate_ranking(top100_candidates)
        print("\n  Local Evaluation Metrics (estimated, not ground truth):")
        print(f"    NDCG@10:              {metrics['ndcg_10']:.4f}")
        print(f"    NDCG@50:              {metrics['ndcg_50']:.4f}")
        print(f"    P@10:                 {metrics['p_10']:.4f}")
        print(f"    Honeypot rate:        {metrics['honeypot_rate']:.4f}")
        print(f"    Estimated composite:  {metrics['estimated_composite']:.4f}")
        print()

    # Optional: run the validator
    if args.validate:
        print("Running validate_submission.py...")
        import subprocess
        validator = Path(__file__).parent.parent / "[PUB] India_runs_data_and_ai_challenge" / \
                    "India_runs_data_and_ai_challenge" / "validate_submission.py"

        if not validator.exists():
            # Try common fallback paths
            fallbacks = [
                Path("../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py"),
                Path("validate_submission.py"),
            ]
            for fb in fallbacks:
                if fb.exists():
                    validator = fb
                    break

        if validator.exists():
            result = subprocess.run(
                [sys.executable, str(validator), str(args.out)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  [OK] {result.stdout.strip()}")
            else:
                print(f"  [WARN] Validation issues:\n{result.stdout}{result.stderr}")
        else:
            print("  [INFO] Validator not found - skipping. Run manually: "
                  "python validate_submission.py submission.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
