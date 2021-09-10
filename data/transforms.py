import torch
from math import sqrt


class BatchwiseStandardizeTransform:
    mean: torch.FloatTensor
    var: torch.FloatTensor
    n: int

    def __init__(self):
        self.mean = 0
        self.var = 0
        self.n = 0

    def __call__(self, x):
        return (x - self.mean) / torch.sqrt(self.var)

    def inverse_transform(self, x):
        return x * torch.sqrt(self.var) + self.mean

    def update(self, x):
        # formulas obtained from http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        n = len(x)
        sample_var, sample_mean = torch.var_mean(x, dim=0, keepdim=True, unbiased=False)
        mu = self.n / (n + self.n) * self.mean + n / (n + self.n) * sample_mean

        var = self.n / (n + self.n) * (self.var + self.mean**2) + n / (n + self.n) * (sample_var + sample_mean**2) - mu**2
        # var = self.n / (n + self.n) * self.var + n / (n + self.n) * sample_var + self.n * n / (self.n+n)**2 * (self.mean - sample_mean)**2
        self.mean = mu
        self.var = var

        self.n += n


class ElementwiseScalarStandardizeTransform(BatchwiseStandardizeTransform):
    mean: float
    var: float

    def __init__(self):
        super(ElementwiseScalarStandardizeTransform, self).__init__()

    def __call__(self, x):
        return (x - self.mean) / sqrt(self.var)

    def inverse_transform(self, x):
        return x * sqrt(self.var) + self.mean

    def update(self, x):
        # formulas obtained from Welford's online algorithm (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
        self.n += 1
        delta = x - self.mean
        self.mean += (x - self.mean) / self.n
        self.var += (delta * (x - self.mean) - self.var) / self.n
