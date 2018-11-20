import numpy as np
class OUNoise:
    def __init__(self, dimension, mu=0.0, theta=0.15, sigma=0.2, seed=123):
        """Initializes the noise """
        self.dimension = dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dimension) * self.mu
        self.reset()
        np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.dimension) * self.mu

    def noise(self) -> np.ndarray:
        x = self.state
        if type(self.dimension) == tuple:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.dimension)
        elif type(self.dimension) == int:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.dimension)
        self.state = x + dx
        return self.state