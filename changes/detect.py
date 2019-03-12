# Inspired by https://github.com/deepcharles/ruptures under the BSD-2 License

from typing import Callable, Tuple, Dict, List, Iterator, Generator

import numpy as np
import scipy.stats
from itertools import cycle
import matplotlib.pyplot as plt

COLOR_CYCLE = ["#4286f4", "#f44174"]
LAMBDA = 100.
ALPHA = 0.1
BETA = 1.
KAPPA = 1.
MU = 0.
THRESHOLD = 0.5


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


def bayesian_online(signal: Iterator, prior_params: dict, changepoint_threshold: float,
                    get_hazard: Callable, get_posterior: Callable) -> Generator[float, None, None]:
    run_start = 0
    run_end = 0
    growth = np.array([1.])
    params = prior_params
    for datum in signal:
        run_length = run_end - run_start
        if len(growth) == run_length + 1:
            growth = np.resize(growth, (run_length + 1) * 2)

        posterior, params = get_posterior(datum, params)
        hazard = get_hazard(np.array(range(run_length + 1)))
        changepoint = np.sum(growth[:run_length + 1] * posterior * hazard)
        #print(np.sum(growth[:run_length + 1]))
        #print(np.sum(posterior))
        #print(np.sum(hazard))
        #print(changepoint)
        growth[1:run_length + 2] = growth[:run_length + 1] * posterior * (1 - hazard)
        growth[0] = changepoint

        growth[:run_length + 2] /= np.sum(growth[:run_length + 2])

        changepoint_detected = changepoint > changepoint_threshold
        if changepoint_detected:
            run_start = run_end
            params = prune_params(run_length - run_start, params)

        run_end += 1
        yield changepoint_detected


def student_posterior(datum: float, params: Dict[str, np.ndarray]
                      ) -> Tuple[float, Dict[str, np.ndarray]]:
    posterior = scipy.stats.t.pdf(x=datum,
                                  df=2 * params['alpha'],
                                  loc=params['mu'],
                                  scale=np.sqrt(params['beta'] * (params['kappa'] + 1) /
                                                (params['alpha'] * params['kappa'])))

    beta = np.concatenate((
        np.array([params['beta'][0]]),
        (params['kappa'] * (datum - params['mu'])**2) / (2 * params['kappa'] + 1)))
    mu = np.concatenate((
        np.array([params['mu'][0]]),
        (params['kappa'] * params['mu'][:1] + datum) / (params['kappa'] + 1)))
    alpha = np.concatenate((
        np.array([params['alpha'][0]]),
        params['alpha'] + 0.5))
    kappa = np.concatenate((
        np.array([params['kappa'][0]]),
        params['kappa'] + 1.0))
    params = {'beta': beta, 'mu': mu, 'alpha': alpha, 'kappa': kappa}
    return posterior, params


def prune_params(cutoff: int, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    params['alpha'] = np.concatenate((params['alpha'][0], params['alpha'][cutoff + 1:]))
    params['beta'] = np.concatenate((params['beta'][0], params['beta'][cutoff + 1:]))
    params['kappa'] = np.concatenate((params['kappa'][0], params['kappa'][cutoff + 1:]))
    params['mu'] = np.concatenate((params['mu'][0], params['mu'][cutoff + 1:]))
    return params


def constant_hazard(steps: np.ndarray, lambda_: float = LAMBDA) -> np.ndarray:
    return np.full_like(a=steps, fill_value=1./lambda_, dtype=np.double)


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
