from torchdiffeq import odeint
import torch

def add_t_arg_to_dynamical_function(f,*args,**kwargs):
    def new_f(t,x):
        return f(x,*args,**kwargs)
    return new_f

def run_odeint_to_final(func,y0,T,inputs=None):
    args = []
    if inputs is not None:
        if len(y0.shape) > 1:
            for _ in range(len(y0.shape) - 1):
                inputs = inputs[None]
            inputs = inputs.repeat(*y0.shape[:-1], 1)
        # print('\nmodified inputs shape',inputs.shape)
        args += [inputs]

    with torch.no_grad():
        traj = odeint(
            add_t_arg_to_dynamical_function(func,*args),
            y0,
            torch.tensor([0,T]).type_as(y0),
        )
    # print('traj shape',traj.shape, 'y0.shape',y0.shape)
    return traj[-1]


if __name__ == "__main__":
    # Test case 1: Simple linear system
    def linear_system(x, inputs=None):
        return -x

    y0 = torch.tensor([1.0])
    T = 1.0
    traj = run_odeint_to_final(linear_system, y0, T)
    print("Linear system test:")
    print("Initial state:", y0)
    print("Final state:", traj[-1])

    # Test case 2: System with external inputs
    def input_system(x, inputs):
        return -x + inputs

    y0 = torch.tensor([1.0])
    T = 1.0
    inputs = torch.tensor([0.5])
    final = run_odeint_to_final(input_system, y0, T, inputs)
    print("\nSystem with inputs test:")
    print("Initial state:", y0)
    print("Input:", inputs)
    print("Final state:", final.shape)

    # Test case 3: Batch processing
    y0_batch = torch.tensor([[1.0], [2.0], [3.0]])
    inputs_batch = torch.tensor([0.5])
    final = run_odeint_to_final(input_system, y0_batch, T, inputs_batch)
    print("\nBatch processing test:")
    print("Initial states:", y0_batch.shape)
    print("Input:", inputs_batch)
    print("Final states:", final.shape)
