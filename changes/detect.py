# Inspired by https://github.com/deepcharles/ruptures under the BSD-2 License

from typing import Callable, Tuple, Dict, List, Iterator

import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

COLOR_CYCLE = ["#4286f4", "#f44174"]


def l1_cost(signal: np.ndarray, weights: np.ndarray) -> float:
    """
    Use the weights on the signal to set the median correctly

    :param signal:
    :param weights:
    :return:
    """
    weighted_differences = (signal - np.median(signal, axis=0)) * weights
    return np.abs(weighted_differences).sum()


def l2_cost(signal: np.ndarray, weights: np.ndarray) -> float:
    """
    Use the weights on the signal to find the standard deviation correctly.

    :param signal:
    :param weights:
    :return:
    """
    weighted_deviation = signal.std(axis=0) * weights
    return (weighted_deviation.mean() * signal.shape[0])**2


def pelt(signal: np.ndarray, weights: np.ndarray, penalty: float,
         cost: Callable[[np.ndarray, np.ndarray], float], jump: int = 2, min_size: int = 2
         ) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Pruned Exact Linear Time (PELT) - https://arxiv.org/pdf/1101.1438.pdf
    for offline changepoint detection.

    The structure of the partitions dictionary is of the form
    {last_end_index: {(start_index, end_index): cost}}

    :param signal:
    :param weights:
    :param jump:
    :param min_size:
    :param penalty:
    :param cost:
    :return:
    """
    n_samples = signal.shape[0]

    # the initial reference partition is the trivial one with the start and end both at 0
    partitions = {0: {(0, 0): 0}}
    starts = []
    # all possible breakpoints for the partition starting at 0.
    ends = [k for k in range(0, n_samples, jump) if k >= min_size] + [n_samples]
    for end in ends:
        start = int(np.floor((end - min_size) / jump) * jump)
        starts.append(start)
        subproblems = []
        for start in starts:
            try:
                left = partitions[start].copy()
            except KeyError:  # no partition of 0:start exists
                continue
            signal_segment = signal[start:end]
            weights_segment = weights[start:end]
            right = {(start, end): cost(signal_segment, weights_segment) + penalty}
            left.update(right)
            subproblems.append(left)

        # pick the partition which ends at end and has the least cost.
        partitions[end] = min(subproblems, key=lambda d: sum(d.values()))
        # trimming the admissible set
        cost_threshold = sum(partitions[end].values()) + penalty
        # only consider the starts which have a cost at least as good as the best found so far.
        starts = [start for start, partition in zip(starts, subproblems)
                  if sum(partition.values()) <= cost_threshold]

    # the best partition contains all of the other partitions.
    best_partition = partitions[n_samples]
    # remove the trivial initial reference partition
    best_partition.pop((0, 0))
    return best_partition


def bocpd(signal: Iterator, get_hazard, observation_likelihood):


    run_start = 0
    run_end = -1
    growth = np.ndarray([1])
    for datum in signal:
        run_end += 1
        run_length = run_end - run_start

        if len(growth) == run_length + 1:
            growth = np.resize(growth, (run_length + 1) * 2)

        predictive = get_pdf(datum)
        hazard = get_hazard(np.array(range(run_length + 1)))
        changepoint = np.sum(growth[:run_length] * predictive * hazard)
        growth[1:run_length + 2] = growth[:run_length + 1] * predictive * (1 - hazard)
        growth[0] = changepoint
        growth[:run_length + 2] /= np.sum(growth[:run_length + 2])


def constant_hazard(lambda_: float, steps: np.ndarray):
    return np.full_like(steps, 1./lambda_)


def plot_breakpoints(signal: np.ndarray, partition: Dict[int, Dict[Tuple[int, int], float]],
                     titles: List[str]) -> None:
    n_samples, n_features = signal.shape
    figsize = (10, 2 * n_features)
    fig, axarr = plt.subplots(nrows=n_features, figsize=figsize, sharex=True)
    if n_features == 1:
        axarr = [axarr]

    for axe, signal_dimension, title in zip(axarr, signal.T, titles):
        color_cycle = cycle(COLOR_CYCLE)
        axe.set_title(title)
        axe.plot(range(n_samples), signal_dimension)

        for (start, end), color in zip(partition.keys(), color_cycle):
            axe.axvspan(max(0, start - 0.5), end - 0.5, facecolor=color, alpha=0.2)

    plt.show()
