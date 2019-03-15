from typing import Iterator, Callable, Generator

import numpy as np
import scipy.stats


def detect_changepoints(signal: Iterator, get_hazard: Callable[[int], np.ndarray],
                        observation_likelihood, delay: int, threshold: float
                        ) -> Generator[bool, None, None]:
    start = 0
    end = 0
    growth_probs = np.array([1.])
    for x in signal:
        run = end - start

        # allocate enough space
        if len(growth_probs) == run + 1:
            growth_probs = np.resize(growth_probs, (run + 1) * 2)

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        hazard = get_hazard(run + 1)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        cp_prob = np.sum(growth_probs[0:run + 1] * pred_probs * hazard)

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        growth_probs[1:run + 2] = growth_probs[0:run + 1] * pred_probs * (1 - hazard)
        # Put back changepoint probability
        growth_probs[0] = cp_prob

        # Renormalize the run length probabilities for improved numerical
        # stability.
        growth_probs[0:run + 2] = growth_probs[0:run + 2] / np.sum(growth_probs[0:run + 2])

        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

        changepoint_detected = run >= delay and growth_probs[delay] >= threshold
        end += 1
        yield changepoint_detected


def constant_hazard(lambda_: float, gap_size: int) -> np.ndarray:
    """Computes the "constant" hazard, that is corresponding
    to Poisson process.
    """
    return np.full(gap_size, 1./lambda_)


class StudentT:
    """Student's t predictive posterior.
    """
    def __init__(self, alpha: float, beta: float, kappa: float, mu: float):
        """
        Initialize the distribution with the priors

        :param alpha:
        :param beta:
        :param kappa:
        :param mu:
        """
        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])

    def pdf(self, data: np.ndarray) -> np.ndarray:
        """
        PDF of the predictive posterior.

        :param data:
        :return:
        """
        return scipy.stats.t.pdf(x=data,
                                 df=2*self.alpha,
                                 loc=self.mu,
                                 scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha * self.kappa)))

    def update_theta(self, data):
        """Bayesian update.
        """
        self.beta = np.concatenate(([self.beta[0]],
                                    self.beta + (self.kappa * (data - self.mu)**2) / (2. * (self.kappa + 1.))))
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
