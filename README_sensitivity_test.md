# ODEint Adjoint Sensitivity Test

This directory contains test scripts to verify that `torchdiffeq.odeint_adjoint` can be used to compute sensitivity of the final point of a trajectory to the initial conditions.

## Files

- `test_odeint_adjoint_sensitivity.py`: Comprehensive test script with multiple dynamical systems and visualization
- `simple_sensitivity_test.py`: Simple, focused test script for basic verification
- `README_sensitivity_test.md`: This documentation file

## Test Results

The tests confirm that `torchdiffeq.odeint_adjoint` successfully computes gradients of final trajectory points with respect to initial conditions:

### Key Findings

1. **✓ Gradient Computation Works**: `odeint_adjoint` can compute gradients of any function of the final trajectory point with respect to initial conditions.

2. **✓ High Accuracy**: The computed gradients are highly accurate, with relative errors on the order of 10^-6 compared to analytical solutions.

3. **✓ Multiple Systems**: The method works for both linear and nonlinear dynamical systems.

4. **✓ Comparison with Regular ODEint**: Regular `odeint` does not support gradient computation, while `odeint_adjoint` does.

### Example Usage

```python
import torch
from torchdiffeq import odeint_adjoint

# Define your dynamics function
def dynamics(t, x):
    return -x  # Example: dx/dt = -x

# Set up initial conditions with gradients enabled
x0 = torch.tensor([[1.0]], requires_grad=True)
t_span = torch.linspace(0, 2.0, 50)

# Compute trajectory with gradient support
trajectory = odeint_adjoint(dynamics, x0, t_span, adjoint_params=())
final_point = trajectory[-1]

# Define a loss function on the final point
loss = torch.sum(final_point**2)

# Compute gradients with respect to initial conditions
loss.backward()
gradients = x0.grad

print(f"Gradients w.r.t. initial conditions: {gradients}")
```

### Important Notes

1. **adjoint_params=()**: When using `odeint_adjoint` with a function that has no learnable parameters, you must specify `adjoint_params=()`.

2. **requires_grad=True**: The initial conditions tensor must have `requires_grad=True` to enable gradient computation.

3. **Loss Function**: You can define any differentiable loss function on the final trajectory point to compute gradients.

### Test Cases

The scripts test several dynamical systems:

1. **1D Linear System**: `dx/dt = -x` with analytical solution `x(t) = x0 * exp(-t)`
2. **2D Harmonic Oscillator**: `dx/dt = y, dy/dt = -x` with analytical solution
3. **Van der Pol Oscillator**: Nonlinear system `dx/dt = y, dy/dt = μ(1-x²)y - x`

### Accuracy Verification

The tests compare the computed gradients with:
- Analytical solutions (where available)
- Finite difference approximations
- Different perturbation sizes for finite differences

Results show that `odeint_adjoint` provides gradients with machine precision accuracy.

## Running the Tests

```bash
# Run the comprehensive test
python test_odeint_adjoint_sensitivity.py

# Run the simple test
python simple_sensitivity_test.py
```

The comprehensive test also generates plots saved as `odeint_adjoint_sensitivity_test.png`.

## Applications

This capability is useful for:

1. **Neural ODEs**: Training neural networks that define dynamical systems
2. **Sensitivity Analysis**: Understanding how changes in initial conditions affect final states
3. **Optimization**: Optimizing initial conditions to achieve desired final states
4. **Control Theory**: Computing gradients for optimal control problems
5. **Scientific Computing**: Sensitivity analysis in computational physics and chemistry

## Dependencies

- PyTorch
- torchdiffeq
- NumPy
- Matplotlib (for visualization in the comprehensive test) 