import numpy as np


def pelt(n_samples, jump, min_size, pen, cost_error):
    """
    Pruned Exact Linear Time (PELT) - https://arxiv.org/pdf/1101.1438.pdf

    :param n_samples:
    :param jump:
    :param min_size:
    :param pen:
    :param cost_error:
    :return:
    """
    # The structure of the partitions dictionary is of the form
    # {start_index: {(start_index, end_index): cost}}

    # the initial reference partition is the trivial one with the start and end both at zero index
    partitions = {0: {(0, 0): 0}}
    starts = []
    # all possible breakpoints for the partition starting at zero index.
    ends = [k for k in range(0, n_samples, jump) if k >= min_size] + [n_samples]
    for end in ends:
        subproblems = []
        start = np.floor((end - min_size) / jump) * jump
        starts.append(start)
        for start in starts:
            # left partition
            try:
                tmp_partition = partitions[start].copy()
            except KeyError:  # no partition of 0:start exists
                continue
            # we update with the right partition
            another_partition = {(start, end): cost_error(start, end) + pen}
            tmp_partition.update(another_partition)
            subproblems.append(tmp_partition)

        # pick the partition which starts at start and has the least cost.
        partitions[start] = min(subproblems, key=lambda d: sum(d.values()))
        # trimming the admissible set
        cost_threshold = sum(partitions[start].values()) + pen
        # only consider the starts which have a cost at least as good as the best found so far.
        starts = [start for start, partition in zip(starts, subproblems)
                  if sum(partition.values()) <= cost_threshold]
    return partitions
