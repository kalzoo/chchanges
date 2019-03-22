from datetime import datetime
from abc import ABC
from typing import Optional, Union

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class Posterior(ABC):

    definition = None

    def pdf(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_theta(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Hazard(ABC):

    definition = None

    def __call__(self, gap: int) -> float:
        raise NotImplementedError


class Detector:

    def __init__(self, hazard: Hazard, posterior: Posterior, delay: int, threshold: float) -> None:
        self.start = 0
        self.end = 0
        self.growth_probs = np.array([1.])
        self.hazard = hazard
        self.posterior = posterior
        self.delay = delay
        self.threshold = threshold

        self.definition = dict(delay=delay, threshold=threshold,
                               **hazard.definition,
                               **posterior.definition)

    def update(self, datum: np.ndarray) -> bool:
        run = self.end - self.start
        self.end += 1

        # allocate enough space
        if len(self.growth_probs) == run + 1:
            self.growth_probs = np.resize(self.growth_probs, (run + 1) * 2)

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = self.posterior.pdf(datum)

        # Evaluate the hazard function for this interval
        hazard_value = self.hazard(run + 1)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        cp_prob = np.sum(self.growth_probs[0:run + 1] * pred_probs * hazard_value)

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        self.growth_probs[1:run + 2] = (self.growth_probs[0:run + 1] *
                                        pred_probs * (1 - hazard_value))
        # Put back changepoint probability
        self.growth_probs[0] = cp_prob

        # Renormalize the run length probabilities for improved numerical
        # stability.
        self.growth_probs[0:run + 2] /= np.sum(self.growth_probs[0:run + 2])

        # Update the parameter sets for each possible run length.
        self.posterior.update_theta(datum)

        changepoint_detected = run >= self.delay and self.growth_probs[self.delay] >= self.threshold
        return changepoint_detected


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def update(self, x_val: Union[float, datetime], y_val: float, yerr: Optional[float] = None):
        self.ax.errorbar(x_val, y_val, yerr=yerr, fmt='k.', alpha=0.3)
        if isinstance(x_val, datetime):
            plt.gcf().autofmt_xdate()
        plt.pause(0.05)


class ConstantHazard(Hazard):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_
        self.definition = {'lambda': lambda_}

    def __call__(self, gap: int) -> np.ndarray:
        """Computes the "constant" hazard, that is corresponding to a Poisson process."""
        return np.full(gap, 1./self.lambda_)


class StudentT(Posterior):
    """Student's t predictive posterior.
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_t.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    """
    def __init__(self, alpha: float, beta: float, kappa: float, mu: float):
        """
        Initialize the distribution with the priors

        :param alpha:
        :param beta:
        :param kappa:
        :param mu:
        """
        self.definition = {'distribution': 'student t', 'alpha': alpha, 'beta': beta,
                           'kappa': kappa, 'mu': mu}
        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])

    def pdf(self, data: np.ndarray) -> np.ndarray:
        """
        PDF of the predictive posterior.
        This needs some real documentation.

        :param data:
        :return:
        """
        return scipy.stats.t.pdf(x=data,
                                 df=2*self.alpha,
                                 loc=self.mu,
                                 scale=np.sqrt(self.beta * (self.kappa+1) /
                                               (self.alpha * self.kappa)))

    def update_theta(self, data):
        """Bayesian update.
        Find some real documentation for this
        """
        self.beta = np.concatenate(([self.beta[0]],
                                    self.beta + (self.kappa * (data - self.mu)**2) /
                                    (2. * (self.kappa + 1.))))
        self.mu = np.concatenate(([self.mu[0]], (self.kappa * self.mu + data) / (self.kappa + 1)))
        self.kappa = np.concatenate(([self.kappa[0]], self.kappa + 1.))
        self.alpha = np.concatenate(([self.alpha[0]], self.alpha + 0.5))

    def prune(self, t):
        """Prunes memory before t.
        """
        self.mu = self.mu[:t + 1]
        self.kappa = self.kappa[:t + 1]
        self.alpha = self.alpha[:t + 1]
        self.beta = self.beta[:t + 1]
