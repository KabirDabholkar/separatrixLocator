import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


# Define the Koopman eigenfunction
psi1 = lambda x: x / np.sqrt(np.abs(1 - x**2))

# Define the bistable dynamics: dx/dt = x - x^3
def bistable_dynamics(x, t):
    return x - x**3

# Time points for integration
t = np.linspace(0, 5, 1000)

# Initial conditions
x0_values = np.linspace(-1.5, 1.5, 10) + 0.05

# Solve the ODE for all initial conditions
x_trajs = []
psi1_trajs = []
for x0 in x0_values:
    x_traj = odeint(bistable_dynamics, x0, t)
    x_trajs.append(x_traj.flatten())
    psi1_traj = psi1(x_traj.flatten())
    psi1_trajs.append(psi1_traj)

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4), sharex=True)

# Plot 1: Dynamics x vs t
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(x0_values)))
for i, (x0, x_traj) in enumerate(zip(x0_values, x_trajs)):
    ax1.plot(t, x_traj, color=colors[i], linewidth=1.5, alpha=0.8)

ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Stable fixed point x = 1')
ax1.axhline(y=-1, color='k', linestyle='--', alpha=0.5, label='Stable fixed point x = -1')
ax1.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Unstable fixed point x = 0')
# ax1.set_xlabel(r'Time $t$')
ax1.set_ylabel(r'$\boldsymbol{x}(t)$')
# ax1.set_title('Dynamics: dx/dt = x - x³')
# ax1.legend()
ax1.spines['left'].set_bounds(-1.7, 1.7)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_bounds(0, 5)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.7, 1.7)
ax1.set_yticks([-1, 0, 1])
ax1.grid(True, alpha=0.3)

# Plot 2: Koopman eigenfunction psi1(x) vs t
for i, (x0, psi1_traj) in enumerate(zip(x0_values, psi1_trajs)):
    ax2.plot(t, np.abs(psi1_traj), color=colors[i], linewidth=1.5, alpha=0.8)

ax2.set_xlabel(r'Time $t$')
ax2.set_ylabel(r'$\big\vert\psi\big(\boldsymbol{x}(t)\big)\big\vert$')
ax2.set_yscale('log')
# ax1.spines['left'].set_bounds(-1, 1)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_bounds(0, 5)
# ax2.set_title('Koopman Eigenfunction ψ₁(x) = x/√(1-x²)')
# ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_for_publication/bistable_kef_with_time.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plots_for_publication/bistable_kef_with_time.png', dpi=300, bbox_inches='tight')
# plt.show()


# Plot psi(x) vs x
fig2, ax = plt.subplots(figsize=(2.5, 2.5))

x_plot = np.linspace(-1.7, 1.7, 1000)
psi_plot = psi1(x_plot)

ax.plot(x_plot, psi_plot, 'k-', linewidth=1.5)
ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
ax.axvline(x=1, color='k', linestyle='--', alpha=0.5) 
ax.axvline(x=-1, color='k', linestyle='--', alpha=0.5)

# Add arrows
ax.arrow(-1.7, 0, 0.3, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax.arrow(-0.3, 0, -0.3, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax.arrow(0.3, 0, 0.3, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax.arrow(1.7, 0, -0.3, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\psi(x)$')
# ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 5)
ax.spines['left'].set_bounds(-4, 4)


plt.tight_layout()
plt.savefig('plots_for_publication/bistable_kef_vs_x.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plots_for_publication/bistable_kef_vs_x.png', dpi=300, bbox_inches='tight')
