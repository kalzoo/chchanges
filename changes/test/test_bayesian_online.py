import numpy as np
from changes.bayesian_online import ConstantHazard, StudentT, Detector


def test_detector():
    normal_signal = np.random.normal(size=1000)
    normal_signal[250:500] += 10.
    normal_signal[500:750] -= 10.
    lambda_ = 100
    delay = 15

    constant_hazard = ConstantHazard(lambda_)
    student_t_observation_likelihood = StudentT(alpha=0.1, beta=1., kappa=1., mu=0.)
    detector = Detector(constant_hazard, student_t_observation_likelihood, delay, threshold=0.5)

    changepoints = []
    for datum in normal_signal:
        changepoints.append(detector.update(datum))

    changepoints = np.argwhere(changepoints).flatten()
    assert len(changepoints) == 3
    expected_changepoints = np.array([250, 500, 750]) + delay - 1
    assert np.linalg.norm(np.array(changepoints) - expected_changepoints, ord=1) < 3
