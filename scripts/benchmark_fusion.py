"""Measure fusion ablation cost on synthetic inputs; this is not an accuracy benchmark."""
import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.perla.merger import LearnCut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--points", type=int, default=768)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compare-inference", action="store_true",
                        help="Compare the previous inference path (with graph losses) against skipping those losses")
    args = parser.parse_args()
    if min(args.batch, args.repeats, args.clusters) < 1 or min(args.queries, args.points) < 24 or args.dim < 2:
        parser.error("batch/repeats must be positive; queries/points >=24; dim >=2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this allocation")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    b, m, n, c = args.batch, args.queries, args.points, args.dim
    inputs = (torch.randn(b, m, c, device=device), torch.rand(b, m, 3, device=device),
              torch.randn(b, n, c, device=device), torch.rand(b, n, 3, device=device),
              torch.randint(16, (b, m), device=device), torch.randint(16, (b, n), device=device))
    context = torch.randn(b, c, device=device)
    variants = (("baseline", "hilbert"), ("baseline", "geometry"), ("gated", "hilbert"),
                ("instruction", "hilbert"), ("instruction", "geometry"))
    baseline_state = None
    baseline_time = None
    baseline_parameters = None
    print(f"Synthetic fusion forward: {args.device}, batch={b}, global={m}, local={n}, dim={c}, K=24")
    if args.compare_inference:
        print("Method         Neighbors   With losses ms  Inference ms  Speedup")
    else:
        print("Method         Neighbors   Median ms  Relative  Extra fusion parameters")
    for fusion, neighbors in variants:
        model = LearnCut(c, c, n_neighs=24, n_clus=args.clusters, n_split=6,
                         fusion_method=fusion, neighbor_selection=neighbors).to(device).eval()
        if baseline_state is None:
            baseline_state = model.state_dict()
        else:
            model.load_state_dict(baseline_state, strict=False)

        def forward(with_losses=True):
            return model(*inputs, instruction_context=context if fusion == "instruction" else None,
                         compute_auxiliary_losses=with_losses)

        timings = []
        inference_timings = []
        with torch.inference_mode():
            for _ in range(3):
                outputs = forward()
            for output in outputs:
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"Non-finite output from {fusion}/{neighbors}")
            if args.compare_inference:
                inference_output = forward(False)
                torch.testing.assert_close(inference_output[0], outputs[0], rtol=0, atol=0)
            for _ in range(args.repeats):
                paths = [(True, timings)]
                if args.compare_inference:
                    paths.append((False, inference_timings))
                for with_losses, measured in paths:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    forward(with_losses)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    measured.append(time.perf_counter() - start)
        median = statistics.median(timings)
        if args.compare_inference:
            inference_median = statistics.median(inference_timings)
            print(f"{fusion:14} {neighbors:10} {median * 1000:14.2f} {inference_median * 1000:13.2f}"
                  f" {median / inference_median:7.2f}x")
            continue
        parameters = sum(p.numel() for p in model.parameters())
        if baseline_time is None:
            baseline_time, baseline_parameters = median, parameters
        print(f"{fusion:14} {neighbors:10} {median * 1000:9.2f} {median / baseline_time:8.2f}x"
              f" {parameters - baseline_parameters:24,d}")
    if args.compare_inference:
        print("Merged features are identical with/without graph losses for every variant.")
    else:
        print("Includes auxiliary graph losses.")
    print("Excludes detector, instruction projection, Q-Former and LLM.")
    print("Synthetic timing does not establish ScanQA/ScanRefer quality or full-model throughput.")


if __name__ == "__main__":
    main()
