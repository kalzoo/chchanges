import numpy as np

from changes.bayesian_online import ConstantHazard, StudentT, Detector, Plotter


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


def run_with_plotter():
    normal_signal = np.random.normal(loc=50e-6, scale=10e-6, size=1000)
    normal_signal[250:500] += 30e-6
    normal_signal[500:750] -= 30e-6
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(alpha=1., beta=1e-12, kappa=1., mu=50e-6, plot=True)
    detector = Detector(hazard, posterior, delay, threshold=0.25)
    value_plotter = Plotter(bottom=0., top=100e-6)
    probability_plotter = Plotter()

    idxs_so_far = []
    for idx, datum in enumerate(normal_signal):
        idxs_so_far.append(idx)
        changepoint_detected = detector.update(datum)
        detector.posterior.update_plot()
        value_plotter.update(idx, datum)
        if idx > delay:
            probability_plotter.update(idx, detector.growth_probs[delay])
        if changepoint_detected:
            changepoint_idx = idxs_so_far[-delay]
            value_plotter.add_changepoint(changepoint_idx)


def variance_run_with_plotter():
    normal_signal1 = np.random.normal(loc=50e-6, scale=10e-6, size=250)
    normal_signal2 = np.random.normal(loc=50e-6, scale=30e-6, size=250)
    normal_signal3 = np.random.normal(loc=50e-6, scale=1e-6, size=250)
    normal_signal4 = np.random.normal(loc=50e-6, scale=10e-6, size=250)
    normal_signal = np.concatenate((normal_signal1, normal_signal2, normal_signal3, normal_signal4))
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(alpha=1., beta=1e-12, kappa=1., mu=50e-6, plot=True)
    detector = Detector(hazard, posterior, delay, threshold=0.25)
    value_plotter = Plotter(bottom=0., top=100e-6)
    probability_plotter = Plotter()

    idxs_so_far = []
    for idx, datum in enumerate(normal_signal):
        idxs_so_far.append(idx)
        changepoint_detected = detector.update(datum)
        detector.posterior.update_plot()
        value_plotter.update(idx, datum)
        if idx > delay:
            probability_plotter.update(idx, detector.growth_probs[delay])
        if changepoint_detected:
            changepoint_idx = idxs_so_far[-delay]
            value_plotter.add_changepoint(changepoint_idx)


if __name__ == "__main__":
    run_with_plotter()
    #variance_run_with_plotter()
    import matplotlib.pyplot as plt
    plt.show()
