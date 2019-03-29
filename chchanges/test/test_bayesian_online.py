import numpy as np

from chchanges.bayesian_online import ConstantHazard, StudentT, Detector


def test_detector():
    # check for small numbers
    normal_signal = np.random.normal(loc=50e-6, scale=1e-6, size=1000)
    normal_signal[250:500] += 30e-6
    normal_signal[500:750] -= 30e-6
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(alpha=1., beta=1e-12, kappa=1., mu=50e-6)
    detector = Detector(hazard, posterior, delay, threshold=0.5)

    changepoints = []
    for datum in normal_signal:
        changepoints.append(detector.update(datum))

    changepoints = np.argwhere(changepoints).flatten()
    assert len(changepoints) == 3
    expected_changepoints = np.array([250, 500, 750]) + delay - 1
    assert np.linalg.norm(np.array(changepoints) - expected_changepoints, ord=1) < 3

    # check for large numbers
    normal_signal = np.random.normal(loc=0., scale=0.1e6, size=1000)
    normal_signal[333:666] += 10e6
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(alpha=1., beta=1e10, kappa=1., mu=50e-6)
    detector = Detector(hazard, posterior, delay, threshold=0.5)

    changepoints = []
    for datum in normal_signal:
        changepoints.append(detector.update(datum))

    changepoints = np.argwhere(changepoints).flatten()
    assert len(changepoints) == 2
    expected_changepoints = np.array([333, 666]) + delay - 1
    assert np.linalg.norm(np.array(changepoints) - expected_changepoints, ord=1) < 2
