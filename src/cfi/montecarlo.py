from statistics import NormalDist

import numpy as np

class MonteCarlo:
    """
    Monte Carlo simulation model for option pricing.
    
    Underlying asset price assumed to follow a geometric Brownian motion.

    """
    def __init__(self, 
                 S: np.ndarray | float | int,
                 T: np.ndarray | float | int,
                 sigma: float,
                 r: float,
                 n: int,
                 q: float = 0,
                 seed: int = 2024,
                ):
        """
        Initializes the Black-Scholes class with the given parameters.

        Parameters:
        S: The spot price of the underlying asset
        T: The time to simulation step (1xM numpy array or scalar)
        sigma: The volatility of the underlying asset
        r: The risk-free interest rate
        n: The number of simulation paths
        q: The continuous dividend yield (default is 0)
        """
                
        if isinstance(S, float) or isinstance(S, int):
            s0 = np.atleast_2d(S)
        if isinstance(T, float) or isinstance(T, int):
            t = np.atleast_2d(T)

        if not all(isinstance(x, np.ndarray) for x in [s0, t]):
            raise ValueError("Expected inputs s0, t to be array or scalar.")
        if any(x <= 0 for x in [sigma, r]):
            raise ValueError("Expected inputs vol, rfr to be greater than 0.")
        if any((x < 0).any() for x in [s0, t]):
            raise ValueError(
                "Expected inputs s0, t to be greater than or equal to 0."
            )

        @property
        def num_periods(self):
            return T.shape[1]

        @property
        def dt(self):
            return np.diff(self.T, prepend=0)

        def generate_paths(self):

            rng = np.random.default_rng(seed=seed)
            
            drift = (self.r - self.q (self.sigma**2) / 2) * self.dt

            dw = self.sigma * rng.standard_normal(size=(self.n, self.num_periods))
            factor = np.multiply(np.sqrt(self.dt), dw)

            return np.multiply(self.S, np.exp(np.cumsum(drift + factor, axis=1)))
