from abc import ABC

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class Posterior(ABC):
    """
    Abstract class defining the interface for the Posterior distribution.

    In the Bayesian Online Changepoint Detection algorithm, the Posterior
        P(x_t | r_{t-1}, x_{t-1}^(r))
    specifies the probability of sampling the next detected data point from the distribution
    associated with the current regime.
    """

    definition = None
    """
    The definition should be overwritten with a dictionary containing the name of the distribution
    as well as its initial parameters.
    """

    def pdf(self, data: np.ndarray) -> np.ndarray:
        """
        Probability density function for the distribution at data.
        If the distribution is d-dimensional, then the data array should have length d.
        If the pruned parameter history has length l, then the returned array should have length l.

        :param data: the data point for which we want to know the probability of sampling from
            the posterior distribution
        :return: the probability of sampling the datapoint from each distribution in the
            pruned parameter history.
        """
        raise NotImplementedError

    def update_theta(self, data: np.ndarray) -> None:
        """
        Use new data to update the posterior distribution.
        The vector of parameters which define the distribution is called theta, hence the name.

        Note that it is important to filter faulty data and outliers before updating theta in
        order to maintain the stability of the distribution.

        :param data: the datapoint which we want to use to update the distribution.
        """
        raise NotImplementedError

    def prune(self, t: int) -> None:
        """
        Remove the parameter history before index t in order to save memory.

        :param t: the index to prune at, e.g. the index of a changepoint.
        """
        raise NotImplementedError


class Hazard(ABC):
    """
    Abstract class defining the interface for the Hazard function.
    """

    definition = None
    """
    The definition should be overwritten with a dictionary containing the name of the hazard 
    function as well as its initial parameters.
    """

    def __call__(self, gap: int) -> float:
        """
        Compute the hazard for a gap between changepoints of a given size.

        :param gap: the number of datapoints since the last changepoint
        :return: the value of the hazard function.
        """
        raise NotImplementedError


class Detector:

    def __init__(self, hazard: Hazard, posterior: Posterior, delay: int, threshold: float):
        """
        Performs Bayesian Online Changepoint Detection as defined in https://arxiv.org/abs/0710.3742

        :param hazard: The hazard provides information on how the occurrence of previous
            changepoints affects the probability of subsequent changepoints.
        :param posterior: The posterior determines the probability of observing a certain data point
            given the data points observed so far.
        :param delay: The delay determines how many data points after a suspected changepoint must
            be measured in order to assure numerical stability.
        :param threshold: the threshold value for considering a changepoint detected,
            somewhat arbitrary, select based on the relative cost of Type 1 vs Type 2 errors
        """
        self.start = 0
        self.end = 0
        self.growth_probs = np.array([1.])
        self.hazard = hazard
        self.posterior = posterior
        self.delay = delay
        self.threshold = threshold

        self.definition = dict(delay=delay, threshold=threshold,
                               hazard=hazard.definition,
                               posterior=posterior.definition)

    def update(self, datum: np.ndarray) -> bool:
        """
        Update the run probabilities based on the new data point and report changepoint if
        the run probability, delayed by self.delay, is greater than self.threshold

        :param datum: the new data point
        :return: Whether a changepoint was detected.
        """
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

    def prune(self) -> None:
        """
        Remove history older than self.delay indices in the past in order to save memory.
        """
        self.posterior.prune(self.delay)
        self.growth_probs = self.growth_probs[:self.delay + 1]
        self.start = self.end - self.delay


class ConstantHazard(Hazard):
    def __init__(self, lambda_: float):
        """
        Computes the constant hazard corresponding to a Poisson process.

        :param lambda_: The average number of indices between events of the Poisson process.
        """
        self.lambda_ = lambda_
        self.definition = {'function': 'constant', 'lambda': lambda_}

    def __call__(self, gap: int) -> np.ndarray:
        """
        Evaluate the hazard function

        :param gap: the number of indices since the last event.
        :return: simply a constant array of length gap.
        """
        return np.full(gap, 1./self.lambda_)


class StudentT(Posterior):
    def __init__(self, var: float, mean: float, df: float = 1., plot: bool = False):
        """
        Student's T predictive posterior.
        https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_t.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t

        Initialize the distribution with initial best guesses for the parameters.

        :param var: A measure of the variance.
        :param mean: The mean of the data collected so far.
        :param df: The number of degrees of freedom
        :param plot: Whether to plot the distribution or not.
        """
        self.definition = {'distribution': 'student t', 'var': var, 'df': df, 'mean': mean}
        self.var = np.array([var])
        self.df = np.array([df])
        self.mean = np.array([mean])

        if plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Distribution over time.")
            self.lines = []

    def pdf(self, data: np.ndarray) -> np.ndarray:
        """
        The probability density function for the Student's T of the predictive posterior.

        Note that t.pdf(x, df, loc, scale) is identically equivalent to t.pdf(y, df) / scale
        with y = (x - loc) / scale. So increased self.var corresponds to increased scale
        which in turn corresponds to a flatter distribution.

        :param data: the data point for which we want to know the probability of sampling from
            the posterior distribution.
        :return: the probability of sampling the datapoint from each distribution in the
            pruned parameter history.
        """
        return scipy.stats.t.pdf(x=data, df=self.df, loc=self.mean,
                                 scale=np.sqrt(2. * self.var * (self.df+1) / self.df ** 2))

    def update_theta(self, data: np.ndarray) -> None:
        """
        Use new data to update the posterior distribution.
        The vector of parameters which define the distribution is called theta, hence the name.

        Note that it is important to filter faulty data and outliers before updating theta in
        order to maintain the stability of the distribution.

        :param data: the datapoint which we want to use to update the distribution.
        """
        next_var = 0.5 * (data - self.mean)**2 * self.df / (self.df + 1.)
        self.var = np.concatenate(([self.var[0]], self.var + next_var))
        self.mean = np.concatenate(([self.mean[0]], (self.df * self.mean + data) / (self.df + 1)))
        self.df = np.concatenate(([self.df[0]], self.df + 1.))

    def prune(self, t: int) -> None:
        """
        Remove the parameter history before index t.

        :param t: the index to prune at, e.g. the index of a changepoint.
        """
        self.mean = self.mean[:t + 1]
        self.df = self.df[:t + 1]
        self.var = self.var[:t + 1]

    def update_plot(self, live: bool = False) -> None:
        """
        Plots the PDF of the distribution based on the latest parameter values

        :param live: If True display the distribution as it evolves, else wait until process stops.
        """
        var = self.var[-1]
        df = self.df[-1]
        mean = self.mean[-1]
        scale = np.sqrt(2. * var * (df + 1) / df**2)
        domain = np.linspace(scipy.stats.t.ppf(0.01, df=df, loc=mean, scale=scale),
                             scipy.stats.t.ppf(0.99, df=df, loc=mean, scale=scale), 100)
        image = scipy.stats.t.pdf(domain, df=df, loc=mean, scale=scale)
        line = self.ax.plot(image, domain, alpha=0.5, color='r')

        # change the previous plots to black
        self.lines.extend(line)
        if len(self.lines) > 1:
            self.lines[-2].set_color('k')
            self.lines[-2].set_alpha(0.03)
        if live:
            plt.pause(0.05)
