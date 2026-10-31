"""Compare base vs new: bitwise output equality + speedup table."""

import json
import sys

import torch

d = sys.argv[1] if len(sys.argv) > 1 else "results"
base = json.load(open(f"{d}/base.json"))
new = json.load(open(f"{d}/new.json"))
ob = torch.load(f"{d}/base_outputs.pt")
on = torch.load(f"{d}/new_outputs.pt")

same = diff = 0
maxdiff = 0.0
for k in ob:
    if torch.equal(ob[k], on[k]):
        same += 1
    else:
        diff += 1
        maxdiff = max(maxdiff, (ob[k] - on[k]).abs().max().item())
print(f"outputs bitwise identical: {same}/{same + diff}  (max abs diff of others: {maxdiff:.3e})")
print(f"worst rel err vs float64 ref: base {base['worst_rel_err']:.3e}  new {new['worst_rel_err']:.3e}")

print("\n| act | M | N | K | G | base ms | new ms | speedup |")
print("|---|---|---|---|---|---|---|---|")
for rb, rn in zip(base["bench"], new["bench"]):
    assert (rb["M"], rb["N"], rb["K"], rb["sym"]) == (rn["M"], rn["N"], rn["K"], rn["sym"])
    print(
        f"| {'sym' if rb['sym'] else 'asym'} | {rb['M']} | {rb['N']} | {rb['K']} | {rb['G']} "
        f"| {rb['ms']:.2f} | {rn['ms']:.2f} | {rb['ms'] / rn['ms']:.1f}x |"
    )
