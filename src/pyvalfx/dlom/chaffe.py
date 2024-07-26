from ..cfi.blackscholes import BlackScholes


class Chaffe:
    def __init__(self, T, sigma, r, q=0):
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

    def calculate_dlom(self):
        return BlackScholes(1, 1, self.T, self.sigma, self.r, self.q).put_price()

    citation = (
        "Chaffe, D.B., '"
        "Option Pricing as a Proxy for Discount for Lack of Marketability in Private Company Valuations"
        "' Business Valuation Review, December 1993, pg 182-188"
    )
