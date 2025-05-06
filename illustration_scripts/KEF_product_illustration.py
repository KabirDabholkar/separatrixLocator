import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 1000)  # Avoid division by zero at x=±1

# Define the functions
psi1 = np.abs(x) / np.sqrt(np.abs(1 - x**2))
psi2 = (x < 0).astype(float)
product = psi1 * psi2

# Shifts
shift1 = 0
shift2 = 6
shift3 = 12

fig, ax = plt.subplots(figsize=(7, 2))

# Plot psi1
ax.plot(x + shift1,0*x,c='grey',ls='dashed')
ax.axvline(shift1,c='grey',ls='dashed')
ax.plot(x + shift1, psi1, color='black', label=r'$\psi_1(x)$')
# Plot psi2
ax.plot(x+shift2,0*x,c='grey',ls='dashed')
ax.axvline(shift2,c='grey',ls='dashed')
ax.plot(x + shift2, psi2, color='black', label=r'$\psi_2(x)$')
# Plot product
ax.plot(x + shift3,0*x,c='grey',ls='dashed')
ax.axvline(shift3,c='grey',ls='dashed')
ax.plot(x + shift3, product, color='black', label=r'$\psi_1(x)\psi_2(x)$')

# Add '×' and '=' symbols between the plots
ymax = np.nanmax(psi1) * 1.05
ax.text(shift2 - 3.1, 1, r'$\times$', fontsize=24, ha='center', va='bottom')
ax.text(shift3 - 3.1, 1, r'$=$', fontsize=24, ha='center', va='bottom')

# Add labels under each plot
# ax.text(shift1, -0.25, r'$\psi_1(x)$', fontsize=16, ha='center')
# ax.text(shift2, -0.25, r'$\psi_2(x)$', fontsize=16, ha='center')
# ax.text(shift3, -0.25, r'$\psi_1(x)\psi_2(x)$', fontsize=16, ha='center')


ax.set_xlim(-2, shift3 + 2)
ax.set_ylim(-0.3, 2) #np.nanmax(psi1) * 1.1)
ax.axis('off')
fig.tight_layout()
fig.savefig('plots_for_publication/product_rule.png', dpi=300, bbox_inches='tight')
fig.savefig('plots_for_publication/product_rule.pdf', dpi=300, bbox_inches='tight')
plt.show()
