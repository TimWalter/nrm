import torch
from tabulate import tabulate

import numpy as np
from scipy.stats import bootstrap


def ci_95(data):
    if len(data) < 2:
        return (-1, -1)
    res = bootstrap((np.array(data),), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low.astype(np.float64), res.confidence_interval.high.astype(np.float64)


import nrm.dataset.r3 as r3
import nrm.dataset.so3 as so3
import nrm.dataset.se3 as se3

from nrm.dataset.morphology import sample_morph
from nrm.dataset.kinematics import inverse_kinematics
from nrm.dataset.reachability_manifold import sample_poses_in_reach, estimate_reachability_manifold
from nrm.logger import binary_confusion_matrix

torch.manual_seed(1)

# Table 1
for level, (interval, n_robots) in enumerate(zip([1, 1, 10, 60], [50, 50, 50, 20])):
    se3.set_level(level + 1)
    print(f"LEVEL {se3.LEVEL}")
    print(f"Fidelity of the discretisation|\t"
          f"# Cells {so3.N_CELLS * (torch.linalg.norm(r3.cell(torch.arange(0, r3.N_CELLS)), dim=1) < 1.0).sum()}|\t"
          f"Distance between neighbouring cells [{se3.MIN_DISTANCE_BETWEEN_CELLS:.3f}, {se3.MAX_DISTANCE_BETWEEN_CELLS:.3f}]\n")

    morphs = sample_morph(n_robots, 6, False)

    coverage = []
    runtime = []
    benchmarks = []
    metrics = []
    batch_size = None
    for morph_idx, morph in enumerate(morphs):
        cell_indices = se3.index(sample_poses_in_reach(100_000, morph))

        _, manipulability = inverse_kinematics(morph.to("cuda"), se3.cell(cell_indices.to(morph.device)).to("cuda"))
        ground_truth = manipulability.cpu() != -1

        coverage += [ground_truth.sum() / ground_truth.shape[0] * 100]
        runtime += [0]

        true_positives = 0.0
        r_indices = torch.empty(0, dtype=torch.int64)
        while true_positives < 95.0 and runtime[-1] < 600:
            new_r_indices, benchmark, batch_size = estimate_reachability_manifold(morph.to("cuda"), True, seconds=interval,
                                                                           batch_size=batch_size)
            r_indices = torch.cat([r_indices, new_r_indices]).unique()
            benchmarks += [torch.tensor(benchmark)]
            runtime[-1] += interval

            labels = torch.isin(cell_indices, r_indices)

            ((true_positives, false_negatives),
             (false_positives, true_negatives)) = binary_confusion_matrix(labels, ground_truth)
        metrics += [[2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100,
                     (ground_truth == labels).sum() / labels.shape[0] * 100,
                     true_positives,
                     false_negatives,
                     false_positives,
                     true_negatives]]

    coverage = torch.tensor(coverage)
    runtime = torch.tensor(runtime)
    benchmarks = torch.stack(benchmarks)
    metrics = torch.tensor(metrics)

    # Mean
    mean_coverage = [coverage.mean().item()]
    mean_runtime = [runtime.float().mean().item()]
    mean_benchmark = benchmarks.mean(dim=0).tolist()
    mean_benchmark[0] = int(mean_benchmark[0])
    mean_benchmark[1] = int(mean_benchmark[1])
    mean_metrics = metrics.mean(dim=0).tolist()
    print("MEAN")
    headers = ["Coverage (%)",
               "Runtime (s)",
               "Filled Cells / 1s",
               "Total Samples / 1s",
               "Efficiency (%)<br>(Total)",
               "Efficiency (%)<br>(Unique)",
               "Efficiency (%)<br>(Collision)",
               "F1 Score (%)",
               "Accuracy (%)",
               "True Positives (%)",
               "False Negatives (%)",
               "False Positives (%)",
               "True Negatives (%)"]
    print(tabulate([mean_coverage + mean_runtime + mean_benchmark + mean_metrics],
                   headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
    # CI
    min_coverage, max_coverage = ci_95(coverage.numpy())
    min_runtime, max_runtime = ci_95(runtime.numpy())
    min_benchmark = []
    max_benchmark = []
    for bench_idx in range(5):
        min_bench, max_bench = ci_95(benchmarks[:, bench_idx].numpy())
        if bench_idx == 0 or bench_idx == 1:
            min_bench = int(min_bench)
            max_bench = int(max_bench)

        min_benchmark += [min_bench]
        max_benchmark += [max_bench]
    min_metrics = []
    max_metrics = []
    for metric_idx in range(6):
        min_metric, max_metric = ci_95(metrics[:, metric_idx].numpy())
        min_metrics += [min_metric]
        max_metrics += [max_metric]
    print("CI Lower")
    print(tabulate([[min_coverage] + [min_runtime] + min_benchmark + min_metrics],
                   headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
    print("CI Upper")
    print(tabulate([[max_coverage] + [max_runtime] + max_benchmark + max_metrics],
                   headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
