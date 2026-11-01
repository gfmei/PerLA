"""Compare repeated scene encoding with shared scene features on a tiny CPU model."""
import argparse
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import torch
from test_captioner import make_captioner_fixture, uncached_densecap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.proposals < 1 or args.repeats < 1:
        parser.error("--proposals and --repeats must be positive")

    torch.set_num_threads(1)
    model, detector, inputs = make_captioner_fixture(args.proposals)

    def reference():
        return uncached_densecap(model, detector, inputs, "dense-cap")

    def optimized():
        return model.predict_densecap(dict(detector), inputs, "dense-cap", max_gen_length=4)

    with torch.no_grad():
        before, after = reference(), optimized()
        torch.testing.assert_close(before[0], after["output_ids"])
        torch.testing.assert_close(before[1], after["l_loss"])
        torch.testing.assert_close(before[2], after["g_loss"])
        timings = {"reference": [], "optimized": []}
        for _ in range(args.repeats):
            for name, function in (("reference", reference), ("optimized", optimized)):
                start = time.perf_counter()
                function()
                timings[name].append(time.perf_counter() - start)

    old = statistics.median(timings["reference"])
    new = statistics.median(timings["optimized"])
    print(f"Synthetic CPU benchmark: batch=2, proposals={args.proposals}, generated_tokens=4")
    print(f"Reference median: {old:.4f}s; optimized median: {new:.4f}s; speedup: {old / new:.2f}x")
    print(f"Scene merges per batch: {args.proposals} -> 1; outputs and losses match")


if __name__ == "__main__":
    main()
