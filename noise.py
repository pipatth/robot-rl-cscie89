import numpy as np

class Noise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=0.02):
        self.mu = mu
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = dt
        self.reset()

    def get_noise(self):
        # compute noise using Euler-Maruyama method
        noise = self.noise_lag1 + self.theta * (self.mu - self.noise_lag1) * \
                self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        
        # set for next round
        self.noise_lag1 = noise
        return noise

    def reset(self):
        self.noise_lag1 = np.zeros_like(self.mu)
