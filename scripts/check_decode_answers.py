#!/usr/bin/env python
"""Do the greedy outputs saved by bench_decode.py (base vs Triton patch) reach the same final answers?

    python scripts/check_decode_answers.py docs/kernels/decode_base.json docs/kernels/decode_patch.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from aha.data import load_eval_dataset  # noqa: E402
from aha.parsing import extract_xml_answer, relaxed_equal  # noqa: E402


def main(base_path: str, patch_path: str) -> None:
    base = json.load(open(base_path))["greedy"]
    patch = json.load(open(patch_path))["greedy"]
    tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-3B-Instruct")
    examples = load_eval_dataset("gsm8k_test", max_samples=len(base))
    same = cb = cp = identical = 0
    for ex, b, p in zip(examples, base, patch):
        ab, ap = extract_xml_answer(tok.decode(b)), extract_xml_answer(tok.decode(p))
        identical += b == p
        same += relaxed_equal(ab, ap)
        cb += relaxed_equal(ab, ex.gold)
        cp += relaxed_equal(ap, ex.gold)
    n = len(base)
    print(f"identical tokens {identical}/{n}; same final answer {same}/{n}; correct base {cb}/{n}, patch {cp}/{n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
