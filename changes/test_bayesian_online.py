import numpy.random
import numpy.linalg
from functools import partial
from changes.bayesian_online import detect_changepoints, constant_hazard, StudentT


def test_bocd():
    normal_signal = numpy.random.normal(size=1000)
    normal_signal[250:500] += 10.
    normal_signal[500:750] -= 10.
    lambda_ = 100
    get_constant_hazard = partial(constant_hazard, lambda_)
    student_t_observation_likelihood = StudentT(alpha=0.1, beta=1., kappa=1., mu=0.)
    changepoints = detect_changepoints(normal_signal, get_constant_hazard,
                                       student_t_observation_likelihood, delay=15, threshold=0.5)
    assert len(changepoints) == 3
    assert numpy.linalg.norm(numpy.array(changepoints) - numpy.array([250, 500, 750]), ord=1) < 3