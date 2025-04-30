import torch
import numpy as np
from functools import partial
from compose import compose

def compose_dynamics(dynamical_function, transform, inverse_transform):
    """
    Composes a dynamical function with coordinate transforms.
    
    Args:
        dynamical_function: Function that takes a tensor and returns its time derivative
        transform: Function that transforms coordinates from original space to new space
        inverse_transform: Function that transforms coordinates from new space back to original space
        
    Returns:
        Function that takes coordinates in new space, transforms to original space,
        applies dynamical function, and transforms back to new space
    """
    composed_dynamics = compose(inverse_transform, dynamical_function, transform)
        
    return composed_dynamics

def change_speed(func,factor=1.0):
    def new_func(*args, **kwargs):
        return func(*args, **kwargs) * factor

    return new_func

def concatenator(functions,split_size_or_sections=2):
    def concatenated_func(z):
        zdots = []
        splits = torch.split(z, split_size_or_sections, dim=-1)
        for func, split in zip(functions, splits):
            zdot = func(split)
            zdots.append(zdot)
        return torch.concat(zdots, axis=-1)
    return concatenated_func

def nonnormal_amplifcation(z):
    A = torch.tensor([[-1,10],[0,-2]]).type_as(z)
    return (z.reshape(-1,z.shape[-1]) @ A.T).reshape(z.shape)


def bistable_ND(z,dim=2,pos=1,scale=1.0):
    mask = torch.arange(dim,device=z.device) == pos
    zdot = (z-z**3) * mask.type(z.dtype) + (-z) * (~mask).type(z.dtype)
    return scale * zdot

A = torch.tensor([[1,1],
                  [0,1]],dtype=torch.float32) #/ np.sqrt(2)


affine_bistable2D = compose_dynamics(
    partial(bistable_ND,dim=2,pos=0),
    lambda x: (A @ x.T).T,
    lambda x: (torch.linalg.inv(A) @ x.T).T,
)

bistable4D_nonnormal = concatenator(
    [partial(bistable_ND, dim=2, pos=0),
    nonnormal_amplifcation],
    split_size_or_sections = 2
)

def bistable_ND_koopman_eigenfunction(z,dim=2,pos=1):
    mask = torch.arange(dim,device=z.device) == pos
    z = (z * mask.type(z.dtype)).sum(axis=--1,keepdims=True)
    return z / ((z**2-1)**2)**0.25

def radial_monostable(z):
    x, y = z[...,0], z[...,1]
    r = torch.sqrt(x**2 + y**2)
    dxdt = x * (1 - r) - y
    dydt = y * (1 - r) + x
    return torch.stack([dxdt, dydt],dim=-1)

def radial_bistable(z):
    x, y = z[...,0], z[...,1]
    r = torch.sqrt(x**2 + y**2)
    dxdt = -x * (r - 1) * (r - 2) - y
    dydt = -y * (r - 1) * (r - 2) + x
    return torch.stack([dxdt, dydt],dim=-1)


def radial_to_cartesian(radial_dynamics):
    """
    Convert a dynamical system in radial coordinates to Cartesian coordinates.

    Parameters:
    - radial_dynamics: function that takes a state (r, θ) and returns
      the dynamics (dr/dt, dθ/dt) in radial coordinates.

    Returns:
    - A function that takes a state (x, y) and returns the dynamics
      (dx/dt, dy/dt) in Cartesian coordinates.
    """

    def cartesian_dynamics(z):
        x, y = z[..., 0], z[..., 1]
        # Compute r and θ from Cartesian coordinates
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.arctan2(y, x)

        # Get dynamics in radial coordinates
        r_theta_dt = radial_dynamics(torch.stack([r, theta],axis=-1))
        dr_dt, dtheta_dt = r_theta_dt[..., 0], r_theta_dt[..., 1]

        # Convert to Cartesian derivatives
        dx_dt = dr_dt * torch.cos(theta) - r * dtheta_dt * torch.sin(theta)
        dy_dt = dr_dt * torch.sin(theta) + r * dtheta_dt * torch.cos(theta)

        return torch.stack([dx_dt, dy_dt], axis=-1)

    return cartesian_dynamics

def radial_bistable(x):
    r,theta = x[...,0:1],x[...,1:2]
    drdt = (r-2) - (r-2)**3
    dthetadt = -torch.sin(theta)
    return torch.concat([drdt, dthetadt],dim=-1)

def radial_bistable_limit_cycle(x):
    r,theta = x[...,0:1],x[...,1:2]
    drdt = (r-2) - (r-2)**3
    dthetadt = x[...,0:1] * 0 + 1 # basically all 1
    return torch.concat([drdt, dthetadt],dim=-1)

def analytical_phi(z,mu = 1.5):
    # r, theta = x[..., 0:1], x[..., 1:2]
    x, y = z[..., 0:1], z[..., 1:2]
    # Compute r and θ from Cartesian coordinates
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.arctan2(y, x)
    # theta_func = torch.abs(torch.sin(theta)**(-1) - torch.tan(theta)**(-1))**(lam-1)
    theta_func = torch.abs(torch.tan(theta/2)) ** (mu - 1)
    # r_func = torch.abs((2-r)/(1-r)/r) ** (mu)
    # r_func = torch.abs((2 - r) / torch.sqrt(torch.abs(r-2**2-4*r+3))) ** (mu)
    r_func = torch.abs((r-2) / torch.sqrt(torch.abs((r-2) ** 2 - 1))) ** (mu)
    return theta_func*r_func

def hopfield(z,A):
    return ( -z + torch.tanh(z) @ (A @ A.T) ) * 5

def init_hopfield(N,R,seed=0,binary=True,normalise=True,scaling=4):
    torch.manual_seed(seed)
    a = torch.randn((N,R))
    if binary:
        a = (a > 1) * 2 - 1
        a = (a).to(torch.float)
    if normalise:
        a /= torch.linalg.norm(a,axis=0,keepdims=True)
    if scaling is not None:
        a *= scaling
    return a

def init_hopfield_ring(N,M,seed=0,binary=True,normalise=True,scaling=4):
    torch.manual_seed(seed)
    theta = torch.arange(0,M) * torch.pi/M
    a = torch.randn((N,2))
    a /= torch.linalg.norm(a, axis=0, keepdims=True)
    a = a @ torch.stack([torch.cos(theta),torch.sin(theta)],axis=0)
    print(theta)

    if binary:
        a = (a > 1) * 2 - 1
        a = (a).to(torch.float)
    if normalise:
        a /= torch.linalg.norm(a,axis=0,keepdims=True)
    if scaling is not None:
        a *= scaling
    return a


if __name__ == '__main__':
    # from functools import partial
    # A = init_hopfield(10,3,seed=0)
    # print(
    #     partial(hopfield,A=init_hopfield(10,3,seed=0))(torch.ones(10,1)).shape
    # )
    # from torchdiffeq import odeint
    # from functools import partial
    # y0 = torch.randn((30,4))
    # times = torch.linspace(0,10,100)
    #
    # func = concatenator(
    #     [partial(bistable_ND, dim=2, pos=0),
    #     nonnormal_amplifcation],
    #     split_size_or_sections = 2
    # )
    # func = change_speed(func,factor=0.1)
    # sol = odeint(lambda t,x: func(x),y0,times)
    # import matplotlib.pyplot as plt
    # plt.plot(sol[...,2],sol[...,3])
    # plt.scatter(sol[-1,..., 2], sol[-1,..., 3],c='red')
    # plt.show()
    # bistable_ND(torch.tensor([0,1])[None,None],dim=2,pos=1)

    from functools import partial
    from compose import compose

    affine_bistable2D = compose(
        lambda x: A @ x,
        partial(
            bistable_ND,
            dim=2,
            pos=0
        ),
        lambda x: np.linalg.inv(A) @ x,
    )


