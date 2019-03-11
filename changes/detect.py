from typing import Callable, Tuple, Dict

import numpy as np


def l1_cost(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Use the weights on the data to set the median correctly

    :param data:
    :param weights:
    :return:
    """
    weighted_differences = (data - np.median(data, axis=0)) * weights
    return np.abs(weighted_differences).sum()


def l2_cost(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Use the weights on the data to find the standard deviation correctly.

    :param data:
    :param weights:
    :return:
    """
    weighted_deviation = data.std(axis=0) * weights
    return (weighted_deviation.mean() * data.shape[0])**2


def pelt(data: np.ndarray, weights: np.ndarray, penalty: float,
         cost: Callable[[np.ndarray, np.ndarray], float], jump: int = 2, min_size: int = 2
         ) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Pruned Exact Linear Time (PELT) - https://arxiv.org/pdf/1101.1438.pdf

    :param data:
    :param weights:
    :param jump:
    :param min_size:
    :param penalty:
    :param cost:
    :return:
    """
    n_samples = data.shape[0]
    # The structure of the partitions dictionary is of the form
    # {start_index: {(start_index, end_index): cost}}

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
            data_segment = data[start:end]
            weights_segment = weights[start:end]
            right = {(start, end): cost(data_segment, weights_segment) + penalty}
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
