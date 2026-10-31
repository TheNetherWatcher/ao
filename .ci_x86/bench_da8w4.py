"""Benchmark + exactness check for torchao::da8w4_linear_cpu.

Usage: python bench_da8w4.py --tag base --outdir results
Writes results/<tag>.json (timings, errors) and results/<tag>_outputs.pt
(raw outputs for bitwise A/B comparison).
"""

import argparse
import json
import os
import statistics
import time

import torch

import torchao  # noqa: F401  registers ops

SHAPES = [
    # (M, N, K, group_size)
    (1, 4096, 4096, 128),
    (4, 4096, 4096, 128),
    (8, 4096, 4096, 128),
    (16, 4096, 4096, 128),
    (32, 4096, 4096, 128),
    (64, 4096, 4096, 128),
    (128, 4096, 4096, 128),
    (256, 4096, 4096, 128),
    (1, 11008, 4096, 32),
    (32, 11008, 4096, 32),
    (128, 4096, 11008, 64),
]

# small odd shapes for correctness only
CHECK_SHAPES = [
    (m, n, k, g)
    for m in (1, 2, 3, 4, 5, 7, 9, 17, 33, 49, 63, 65, 97, 130)
    for (n, k, g) in ((32, 64, 32), (64, 256, 128), (96, 512, 256), (32, 128, 64))
]


def make(M, N, K, G, sym, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 16, (N, K), dtype=torch.uint8, generator=g)
    sb = torch.rand(N, K // G, generator=g) * 0.02 + 0.001
    zb = torch.randint(0, 16, (N, K // G), dtype=torch.int8, generator=g)
    if sym:
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, generator=g)
        za = torch.zeros(M, dtype=torch.int32)
    else:
        a = torch.randint(0, 256, (M, K), dtype=torch.uint8, generator=g)
        za = torch.randint(0, 256, (M,), dtype=torch.int32, generator=g)
    sa = torch.rand(M, generator=g) * 0.05 + 0.001
    bias = torch.randn(N, generator=g)
    return a, sa, za, w, sb, zb, bias


def reference(a, sa, za, w, sb, zb, bias, G):
    M, K = a.shape
    N = w.shape[0]
    ad = a.to(torch.float64) - za.to(torch.float64)[:, None]  # [M, K]
    zbk = zb.to(torch.float64).repeat_interleave(G, dim=1)  # [N, K]
    wd = w.to(torch.float64) - zbk  # [N, K]
    sbk = sb.to(torch.float64).repeat_interleave(G, dim=1)
    y = (ad @ (wd * sbk).T) * sa.to(torch.float64)[:, None]
    return y + bias.to(torch.float64)


def run(args_, packed, bias):
    a, sa, za = args_
    pw, ps, pz, comp = packed
    return torch.ops.torchao.da8w4_linear_cpu.default(
        a, sa, za, pw, ps, pz, comp, bias, torch.float
    )


def timeit(fn, min_time=1.0, min_reps=5):
    fn()
    times = []
    t_start = time.perf_counter()
    while len(times) < min_reps or time.perf_counter() - t_start < min_time:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 200:
            break
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default="results")
    p.add_argument("--no-bench", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("threads", torch.get_num_threads())
    res = {"tag": args.tag, "threads": torch.get_num_threads(), "check": [], "bench": []}
    outputs = {}

    worst = 0.0
    for sym in (True, False):
        for M, N, K, G in CHECK_SHAPES + (SHAPES if not args.no_bench else []):
            a, sa, za, w, sb, zb, bias = make(M, N, K, G, sym, seed=M * 7 + N + K + G)
            packed = torch.ops.torchao.da8w4_linear_prepack_cpu.default(w, sb, zb)
            for b in (None, bias):
                y = run((a, sa, za), packed, b)
                ref = reference(a, sa, za, w, sb, zb, b if b is not None else torch.zeros(N), G)
                err = ((y.double() - ref).abs() / (ref.abs() + 1e-3)).max().item()
                worst = max(worst, err)
                key = f"{'sym' if sym else 'asym'}_{M}x{N}x{K}_g{G}_{'bias' if b is not None else 'nobias'}"
                outputs[key] = y
                res["check"].append({"key": key, "max_rel_err": err})
    res["worst_rel_err"] = worst
    print(f"worst relative error vs float64 reference: {worst:.3e}")
    torch.save(outputs, os.path.join(args.outdir, f"{args.tag}_outputs.pt"))

    if not args.no_bench:
        for sym in (True, False):
            for M, N, K, G in SHAPES:
                a, sa, za, w, sb, zb, bias = make(M, N, K, G, sym, seed=0)
                packed = torch.ops.torchao.da8w4_linear_prepack_cpu.default(w, sb, zb)
                t = timeit(lambda: run((a, sa, za), packed, bias))
                gops = 2 * M * N * K / t / 1e9
                row = {"sym": sym, "M": M, "N": N, "K": K, "G": G, "ms": t * 1e3, "gops": gops}
                res["bench"].append(row)
                print(f"{'sym ' if sym else 'asym'} M={M:4d} N={N:5d} K={K:5d} G={G:3d}  {t*1e3:9.3f} ms  {gops:8.2f} GOP/s", flush=True)

    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)

    assert worst < 1e-4, f"error too large: {worst}"


if __name__ == "__main__":
    main()
