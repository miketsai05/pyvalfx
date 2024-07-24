"""
Module for Black Scholes formula
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S, K, T, sigma, r, q=0):
        """
        Initializes the Black-Scholes class with the given parameters.

        Parameters:
        S: The spot price of the underlying asset
        K: The strike price
        T: The time to maturity (in years)
        sigma: The volatility of the underlying asset
        r: The risk-free interest rate
        q: The continuous dividend yield (default is 0)
        """

        if any(x <= 0 for x in [T, sigma, r]):
            raise ValueError("Expected inputs T, sigma, rfr to be greater than 0")
        if any(x < 0 for x in [S, K]):
            raise ValueError("Expected inputs S, K to be greater than or equal to 0")

        self.S = np.asarray(S)
        self.K = np.asarray(K)
        self.T = np.asarray(T)
        self.sigma = np.asarray(sigma)
        self.r = np.asarray(r)
        self.q = np.asarray(q)

        # Check if all non-scalar inputs have the same dimensions
        shapes = [x.shape for x in [self.S, self.K, self.T, self.sigma, self.r, self.q] if x.ndim > 0]
        if len(set(shapes)) > 1:
            raise ValueError("All non-scalar inputs must have the same dimensions.")

    @property
    def d1(self):
        """Calculates d1 values in  Black-Scholes formula"""
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    @property
    def d2(self):
        """Calculates d2 value in Black-Scholes formula"""
        return self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """
        Calculates the price of a European call option using the Black-Scholes formula.
        """
        return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(
            self.d2
        )

    def put_price(self):
        """
        Calculates the price of a European put option using the Black-Scholes formula.
        """
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(
            -self.d1
        )
