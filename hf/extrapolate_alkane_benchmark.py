"""Two-point power-law extrapolation of the alkane one-pass benchmark."""

import json
import math
import sys
from pathlib import Path


TARGETS = (
    ("C517H1036", 30014),
    ("C862H1726", 50024),
)


def fit(size_a, value_a, size_b, value_b):
    if value_a <= 0 or value_b <= 0 or size_a <= 0 or size_b <= 0 or size_a == size_b:
        return None
    exponent = math.log(value_b / value_a) / math.log(size_b / size_a)
    scale = value_a / (size_a ** exponent)
    return scale, exponent


def predict(scale_exponent, size):
    if scale_exponent is None:
        return None
    scale, exponent = scale_exponent
    return scale * size ** exponent


def load(path):
    data = json.loads(Path(path).read_text())
    stages = {stage["stage"]: stage for stage in data["stages"]}
    return int(data["basis_functions"]), stages


def main(argv):
    if len(argv) != 3:
        print("usage: extrapolate_alkane_benchmark.py small.json large.json", file=sys.stderr)
        return 2
    size_a, stages_a = load(argv[1])
    size_b, stages_b = load(argv[2])
    if size_a > size_b:
        size_a, stages_a, size_b, stages_b = size_b, stages_b, size_a, stages_a
    names = [name for name in stages_a if name in stages_b]
    print(f"measured_basis_functions: {size_a} {size_b}")
    print("fit: value = scale * nao ** exponent from two points; no third point checks the residual")
    projections = []
    for name in names:
        for metric in ("seconds", "output_gib", "rss_increase_gib"):
            fitted = fit(
                size_a, stages_a[name][metric],
                size_b, stages_b[name][metric],
            )
            for label, target in TARGETS:
                value = predict(fitted, target)
                exponent = None if fitted is None else fitted[1]
                projections.append((name, metric, label, target, value, exponent))
                shown = "undefined" if value is None else f"{value:.6g}"
                exponent_text = "undefined" if exponent is None else f"{exponent:.4f}"
                print(
                    f"predict stage={name} metric={metric} system={label} "
                    f"basis_functions={target} value={shown} exponent={exponent_text}"
                )
    time_rows = [row for row in projections if row[1] == "seconds" and row[4] is not None]
    memory_rows = [row for row in projections if row[1] == "output_gib" and row[4] is not None]
    for label, _target in TARGETS:
        slowest = max((row for row in time_rows if row[2] == label), key=lambda row: row[4])
        largest = max((row for row in memory_rows if row[2] == label), key=lambda row: row[4])
        print(
            f"bottleneck system={label} slowest_stage={slowest[0]} "
            f"predicted_seconds={slowest[4]:.6g} "
            f"largest_memory_stage={largest[0]} "
            f"predicted_output_gib={largest[4]:.6g}"
        )
    print("下一步是制定优化方案。这次只标出瓶颈，不改求解器。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
