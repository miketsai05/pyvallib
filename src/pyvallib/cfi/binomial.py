import numpy as np


class BinomialCRR:
    """
    Cox-Ross-Rubinstein binomial lattice model for option pricing with the given parameters.

    Parameters:
    S: The spot price of the underlying asset
    T: The total time horizon from valuation date
    sigma: The volatility of the underlying asset
    r: The risk-free rate
    M: The number of time steps
    q: The continuous dividend yield (default is 0)

    Underlying asset price assumed to follow a geometric Brownian motion.

    Generates shape-(M+1xM+1) array lattice where:
    M is the number of metric periods

    """

    def __init__(
        self,
        S: float,
        T: float,
        sigma: float,
        r: float,
        M: int,
        q: float = 0,
    ):
        self.S = S
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self.M = int(M)

    @property
    def dt(self):
        """Length of time for each time step"""
        return self.T / self.M

    @property
    def u(self):
        """Up factor"""
        return np.exp(self.sigma * np.sqrt(self.dt))

    @property
    def d(self):
        """Down factor"""
        return 1 / self.u

    @property
    def p_u(self):
        """Probability of up movement"""
        return (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    @property
    def p_d(self):
        """Probability of down movement"""
        return 1 - self.p_u

    def generate_lattice(self):
        """Generate lattice of underlying stock prices"""
        lattice = np.zeros((self.M + 1, self.M + 1))
        lattice[0, 0] = self.S

        for i in range(1, self.M + 1):
            lattice[:i, i] = lattice[:i, i - 1] * self.u
            lattice[i, i] = lattice[i - 1, i - 1] * self.d

        return lattice

    def rollback_lattice(self, payoff_func, rollback_func):
        """
        Rollback lattice to calculate option price given the following parameters.

        Parameters:
        payoff_func: The payoff function of the option at maturity
        rollback_func: The function evaluated at each node to compare early exercise value vs the continuation value
        """
        stock_lattice = self.generate_lattice()
        option_lattice = np.zeros(stock_lattice.shape)
        option_lattice[:, -1] = payoff_func(stock_lattice[:, -1])
        for i in reversed(range(1, self.M + 1)):
            option_lattice[:i, i - 1] = np.maximum(
                rollback_func(stock_lattice[:i, i - 1]),
                np.exp(-self.r * self.dt)
                * (self.p_u * option_lattice[:i, i] + self.p_d * option_lattice[1 : i + 1, i]),
            )
        return option_lattice

    # INCUDE OPTIONAL INPUT/METHOD TO ASSOCIATE DATES WITH NODES WITHIN CLASS

    citation = (
        "Cox, J.; Ross, S.; Rubinstein, M. (1979) 'Option Pricing: A Simplified Approach', "
        "Journal of Financial Economics, 7(3), pp. 229-263."
    )


class BinomialAmerican(BinomialCRR):
    """
    Cox-Ross-Rubinstein binomial lattice model for American call/put pricing with the given parameters.

    Parameters:
    S: The spot price of the underlying asset
    K: The strike price
    T: The total time horizon from valuation date
    sigma: The volatility of the underlying asset
    r: The risk-free interest rate
    M: The number of time steps
    q: The continuous dividend yield (default is 0)

    Underlying asset price assumed to follow a geometric Brownian motion.

    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float,
        M: int,
        q: float = 0,
    ):
        super().__init__(S, T, sigma, r, M, q)
        self.K = K

    def call_price(self):
        """
        Calculates the price of an American call option using the binomial lattice model.
        """

        def payoff_func(x):
            return np.maximum(x - self.K, 0)

        return self.rollback_lattice(payoff_func, payoff_func)[0, 0]

    def put_price(self):
        """
        Calculates the price of an American put option using the binomial lattice model.
        """

        def payoff_func(x):
            return np.maximum(self.K - x, 0)

        return self.rollback_lattice(payoff_func, payoff_func)[0, 0]
