"""Ask one question from the command line.

    python -m healthrag.ask "What does the hypertension guideline recommend first?"
    python -m healthrag.ask "What medications is the patient taking?" --patient 00e408ae605a49c6
"""

from __future__ import annotations

import argparse
import logging
import sys

from healthrag.config import get_settings
from healthrag.generate.factory import build_answerer
from healthrag.retrieve.filters import Filters


def main(argv: list[str] | None = None) -> int:
    """Answer a question and print the answer, citations, route and timings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--patient", help="pseudonymous patient_id to restrict retrieval to")
    parser.add_argument("--doc-type", choices=["patient_record", "guideline"])
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--router", choices=["rules", "rules_llm", "trained"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # Windows consoles

    answerer = build_answerer(get_settings(), args.router)
    filters = Filters(args.doc_type, args.patient, args.date_from, args.date_to)
    answer = answerer.answer(args.question, filters)
    print(answer.text)
    print()
    print(f"route: {answer.route.tier} ({answer.route.router}: {answer.route.reason}) -> {answer.model}")
    print(f"citations: {', '.join(answer.citations) or 'none'}")
    if answer.invalid_citations:
        print(f"invalid citations (not retrieved): {', '.join(answer.invalid_citations)}")
    print(f"answer PHI hits: {answer.answer_phi_hits}")
    print(f"tokens: {answer.tokens_in} in / {answer.tokens_out} out, est. ${answer.estimated_cost_usd:.6f}")
    print("latency ms: " + ", ".join(f"{k}={v:.0f}" for k, v in answer.timings_ms.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
