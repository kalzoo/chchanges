import numpy as np
from matplotlib import pyplot as plt

from chchanges.bayesian_online import ConstantHazard, StudentT, Detector


def detect_variance_shift():
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

    _, data_axis = plt.subplots()
    data_axis.set_title('Data Stream')
    data_axis.set_xlabel('Datum index')
    data_axis.set_ylabel('Datum value')

    _, prob_axis = plt.subplots()
    prob_axis.set_title('Probability Stream')
    prob_axis.set_xlabel('Datum index')
    prob_axis.set_ylabel('Probability of changepoint')

    idxs_so_far = []
    for idx, datum in enumerate(normal_signal):
        idxs_so_far.append(idx)
        changepoint_detected = detector.update(datum)
        detector.posterior.update_plot()
        data_axis.plot(idx, datum, color='k', alpha=0.15)
        if idx > delay:
            prob_axis.plot(idx, detector.growth_probs[delay], color='k', alpha=0.15)
        if changepoint_detected:
            changepoint_idx = idxs_so_far[-delay]
            data_axis.axvline(changepoint_idx, alpha=0.5, color='r', linestyle='--')
        plt.pause(0.05)
    plt.show()


if __name__ == '__main__':
    detect_variance_shift()
