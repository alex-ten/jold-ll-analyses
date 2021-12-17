# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
from seaborn import kdeplot

# %%
def mutual_info1(xy, kxy, kx, ky):
    p_xy = kxy.pdf(xy.T)
    # logp_xy = kxy.logpdf(xy.T)
    p_x, p_y = kx.pdf(xy.T[0]), ky.pdf(xy.T[1])
    # logp_x, logp_y = kx.logpdf(xy.T[0]), ky.logpdf(xy.T[1])
    like_ratio = np.log(p_xy / np.clip(p_x * p_y, a_min=1e-6, a_max=1e6))
    like_ratio = np.log(p_xy / np.clip(p_x * p_y, a_min=1e-8, a_max=1e8))
    log_like_ratio = np.clip(like_ratio, a_min=0, a_max=1e6)
    return np.sum(p_xy * log_like_ratio)


def sample_linear(x, slope, noise):
    y = slope * x + np.random.normal(0, noise, size=x.size)
    return np.stack([x, y], axis=1)

# %%
np.random.seed(1)

x = np.random.normal(0, 1, 300)
xy0 = sample_linear(x, .00, 1)
xy1 = sample_linear(x, .25, 1)
xy2 = sample_linear(x, .50, 1)

k0 = gaussian_kde(xy0.T)
kx0, ky0 = gaussian_kde(xy0[0]), gaussian_kde(xy0[1])

k1 = gaussian_kde(xy1.T)
kx1, ky1 = gaussian_kde(xy1[0]), gaussian_kde(xy1[1])

k2 = gaussian_kde(xy2.T)
kx2, ky2 = gaussian_kde(xy2[0]), gaussian_kde(xy2[1])

fig, ax = plt.subplots(num='samples', nrows=1, ncols=3, figsize=[10, 3])
ax[0].scatter(xy0[:, 0], xy0[:, 1], s=5, alpha=.2)
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)
ax[1].scatter(xy1[:, 0], xy1[:, 1], s=5, alpha=.2)
ax[1].set_xlim(-3, 3)
ax[1].set_ylim(-3, 3)
ax[2].scatter(xy2[:, 0], xy2[:, 1], s=5, alpha=.2)
ax[2].set_xlim(-3, 3)
ax[2].set_ylim(-3, 3)

fig, ax = plt.subplots(num='distrs', nrows=1, ncols=3, figsize=[10, 3])
kdeplot(x=xy0[:, 0], y=xy0[:, 1], fill=True, ax = ax[0])
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)
kdeplot(x=xy1[:, 0], y=xy1[:, 1], fill=True, ax = ax[1])
ax[1].set_xlim(-3, 3)
ax[1].set_ylim(-3, 3)
kdeplot(x=xy2[:, 0], y=xy2[:, 1], fill=True, ax = ax[2])
ax[2].set_xlim(-3, 3)
ax[2].set_ylim(-3, 3)

print(f'Mutual information with covar: {mutual_info1(xy0, k0, kx0, ky0)}')
print(f'Mutual information with covar: {mutual_info1(xy1, k1, kx1, ky1)}')
print(f'Mutual information with covar: {mutual_info1(xy2, k2, kx2, ky2)}')
# %%
