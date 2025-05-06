import numpy as np
from plotting import evaluate_on_grid, plot_flow_streamlines, remove_frame
from functools import partial
import matplotlib.pyplot as plt
import torch

def dynamics(z):
    return z-z**3

def psi1D(x):
    return torch.abs(x) / torch.sqrt(torch.abs(x-x**3))

def psi2D(z,mu=0.5):
    x,y = z[...,0],z[...,1]
    return  (psi1D(x) ** mu + psi1D(y) ** (1-mu))


# fig,ax = plt.subplots(1,1,figsize=(4,4))
# X,Y,psi = evaluate_on_grid(partial(psi2D,mu=0.5),x_limits=(-1.5,1.5),y_limits=(-1.5,1.5))
# ax.contourf(X, Y, psi, levels=15, cmap='Blues_r')
# plt.show()

x_limits = (-1.5, 1.5)
y_limits = (-1.5, 1.5)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

mus = [0, 0.5, 1]
eps = 1
for ax, mu in zip(axs, mus):
    X, Y, psi = evaluate_on_grid(partial(psi2D, mu=mu), x_limits=x_limits, y_limits=y_limits, resolution=200)
    abs_psi = psi
    abs_psi[abs_psi>3] = np.inf
    if mu == 0:
        signed_psi = np.sign(Y) * psi
    elif mu == 1:
        signed_psi = np.sign(X) * psi
    else:
        signed_psi = np.sign(X) * np.sign(Y) * psi

    ax.contourf(X, Y, abs_psi, levels=15, cmap='Blues_r')
    CS = ax.contour(X, Y, signed_psi, levels=[0], colors='red', linewidths=4)
    ax.clabel(CS, CS.levels, fontsize=10)
    ax.set_title(f'μ = {mu}')
    plot_flow_streamlines(dynamics,ax,x_limits=x_limits, y_limits=y_limits, resolution=25, color='green', alpha=0.5, density=0.6)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    remove_frame(ax)

fig.tight_layout()
plt.show()
fig.savefig('../plots_for_publication/KEF_degeneracy.pdf')
fig.savefig('../plots_for_publication/KEF_degeneracy.png', dpi=300)


