import numpy as np


class MonteCarlo:
    """
    Monte Carlo simulation model for option pricing with the given parameters.

    Parameters:
    S: The spot price of the underlying asset
    T: The time to simulation step from valuation date (1xM numpy array or scalar)
    sigma: The volatility of the underlying asset
    r: The risk-free interest rate
    n: The number of simulation paths
    q: The continuous dividend yield (default is 0)
    seed: The seed for the random number generator (default is 2024)

    Underlying asset price assumed to follow a geometric Brownian motion.

    Generates shape-(nxM) array of simulation paths where:
    n is the number of simulation paths
    M is the number of metric periods

    """

    def __init__(
        self,
        S: float,
        T: np.ndarray | float,
        sigma: float,
        r: float,
        n: int,
        q: float = 0,
        seed: int = 2024,
    ):
        if isinstance(T, (float, int)):
            T = np.atleast_2d(T)
        if not isinstance(T, np.ndarray):
            raise ValueError("Expected input T to be array or scalar")

        if any(x <= 0 for x in [S, *T, sigma, r]):
            raise ValueError("Expected inputs S, T, sigma, rfr to be greater than 0")

        self.S = S
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n = int(n)
        self.seed = seed

    @property
    def M(self):
        """Number of simulation steps"""
        return self.T.shape[1]

    @property
    def dt(self):
        return np.diff(self.T, prepend=0)

    def generate_paths(self):
        rng = np.random.default_rng(seed=self.seed)

        drift = ((self.r - self.q) - (self.sigma**2) / 2) * self.dt

        dw = self.sigma * rng.standard_normal(size=(self.n, self.M))
        factor = np.multiply(np.sqrt(self.dt), dw)

        return np.multiply(self.S, np.exp(np.cumsum(drift + factor, axis=1)))
