import numpy as np
from matplotlib import pyplot as plt

from chchanges.bayesian_online import ConstantHazard, StudentT, Detector


def detect_mean_shift():
    normal_signal = np.random.normal(loc=50e-6, scale=10e-6, size=1000)
    normal_signal[250:500] += 30e-6
    normal_signal[500:750] -= 30e-6
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(var=1e-12, df=1., mean=50e-6, plot=True)
    detector = Detector(hazard, posterior, delay, threshold=0.25)

    data_plotter_fig, data_plotter_ax = plt.subplots()
    data_plotter_ax.set_title('Data Stream')
    data_plotter_ax.set_xlabel('Datum index')
    data_plotter_ax.set_ylabel('Datum value')

    prob_plotter_fig, prob_plotter_ax = plt.subplots()
    prob_plotter_ax.set_title('Probability Stream')
    prob_plotter_ax.set_xlabel('Datum index')
    prob_plotter_ax.set_ylabel('Probability of changepoint')

    idxs_so_far = []
    for idx, datum in enumerate(normal_signal):
        idxs_so_far.append(idx)
        changepoint_detected = detector.update(datum)
        detector.posterior.update_plot(live=True)
        data_plotter_ax.errorbar(idx, datum, fmt='k.', alpha=0.3)
        if idx > delay:
            prob_plotter_ax.errorbar(idx, detector.growth_probs[delay], fmt='k.', alpha=0.3)
        if changepoint_detected:
            changepoint_idx = idxs_so_far[-delay]
            data_plotter_ax.axvline(changepoint_idx, alpha=0.5, color='r', linestyle='--')
    plt.show()


if __name__ == '__main__':
    detect_mean_shift()
