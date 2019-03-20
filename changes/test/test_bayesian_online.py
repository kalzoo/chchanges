import numpy as np
from changes.bayesian_online import ConstantHazard, StudentT, Detector


def test_detector():
    normal_signal = np.random.normal(loc=50e-6, scale=1e-6, size=1000)
    normal_signal[250:500] += 150e-6
    normal_signal[500:750] -= 150e-6
    lambda_ = 250
    delay = 200

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(alpha=1., beta=1e-6, kappa=1., mu=50e-6)
    detector = Detector(hazard, posterior, delay, threshold=0.2)

    changepoints = []
    for datum in normal_signal:
        changepoints.append(detector.update(datum))

    changepoints = np.argwhere(changepoints).flatten()
    assert len(changepoints) == 3
    expected_changepoints = np.array([250, 500, 750]) + delay - 1
    assert np.linalg.norm(np.array(changepoints) - expected_changepoints, ord=1) < 3
