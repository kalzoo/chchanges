import numpy as np

from changes.detect import (bayesian_online, constant_hazard, student_posterior,
                            ALPHA, BETA, KAPPA, MU, THRESHOLD)

DELAY = 15


def test_bayesian_online():
    signal = np.random.normal(size=1000)
    signal[250:500] += 10
    signal[500:750] -= 10
    prior_params = {'alpha': np.array([ALPHA]),
                    'beta': np.array([BETA]),
                    'kappa': np.array([KAPPA]),
                    'mu': np.array([MU])}
    changepoint_stream = bayesian_online(signal, prior_params, THRESHOLD, constant_hazard,
                                         student_posterior)
    changepoints = list(changepoint_stream)
    assert len(changepoints) == 3
    assert np.linalg.norm(np.array(changepoints) - np.array([250, 500, 750]), ord=1) < 3
