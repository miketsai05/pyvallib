# import numpy as np
# from .binomial import BinomialCRR

# class ConvertibleTF(BinomialCRR):
#     """
#     Tsiveriotis-Fernandes lattice model for convertible bonds with the given parameters.

#     Parameters:
#     S: The spot price of the underlying asset
#     conv_price = The conversion price
#     T: The total time horizon from valuation date
#     sigma: The volatility of the underlying asset
#     r: The risk-free rate
#     r_risky: The risky rate
#     M: The number of time steps
#     q: The continuous dividend yield (default is 0)

#     Underlying asset price assumed to follow a geometric Brownian motion.

#     Generates shape-(M+1xM+1) array lattice where:
#     M is the number of metric periods

#     """

#     def __init__(
#         self,
#         S: float,
#         conv_price: float,
#         T: float,
#         sigma: float,
#         r: float,
#         r_risky: float,
#         M: int,
#         q: float = 0,
#     ):
#         super().__init__(S, T, sigma, r, M, q)
#         self.conv_price = conv_price
#         self.r_risky = r_risky

#     def rollback_lattice(self, payoff_func, rollback_func):
#         stock_lattice = self.generate_lattice()
#         bond_lattice = np.zeros_like(stock_lattice)
#         convertible_lattice = np.zeros_like(stock_lattice)
