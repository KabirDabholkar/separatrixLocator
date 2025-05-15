import numpy as np

def cubic_hermite(x, y, m_x, m_y, num_points=100,lims=[0,1]):
    """
    Generate points on a cubic Hermite curve joining points x and y with tangents m_x and m_y.

    Arguments:
    - x: Starting point of the curve.
    - y: Ending point of the curve.
    - m_x: Tangent vector at the starting point.
    - m_y: Tangent vector at the ending point.
    - num_points: Number of points to generate on the curve.

    Returns:
    - points: Array of points on the cubic Hermite curve.
    """
    alpha = np.linspace(lims[0], lims[1], num_points)

    # Cubic Hermite interpolation formula
    points = (2 * alpha ** 3 - 3 * alpha ** 2 + 1)[:, np.newaxis] * x + \
             (-2 * alpha ** 3 + 3 * alpha ** 2)[:, np.newaxis] * y + \
             (alpha ** 3 - 2 * alpha ** 2 + alpha)[:, np.newaxis] * m_x + \
             (alpha ** 3 - alpha ** 2)[:, np.newaxis] * m_y

    return points

def generate_curves_between_points(x, y, num_curves=100, num_points=100, rand_scale=3.0, max_tries=1000, lims=[0.01,0.99], return_alpha=False, non_negative=False):
    """
    Generate multiple curves between two points using cubic Hermite interpolation.
    
    Args:
        x (np.ndarray): Starting point
        y (np.ndarray): Ending point
        num_curves (int): Number of curves to generate
        num_points (int): Number of points per curve
        rand_scale (float): Scale factor for random perturbations of tangent vectors
        max_tries (int): Maximum number of attempts to find a valid curve
        non_negative (bool): If True, ensures all points on curves are non-negative
        
    Returns:
        np.ndarray: Array of shape (num_curves, num_points, dim) containing all valid curves
    """
    # Calculate base tangent vectors
    m_x = -x + y  # Tangent at x
    m_y = -x + y  # Tangent at y
    
    all_points = []
    for _ in range(num_curves):
        # Generate points until we get a valid curve
        num_tries = 0
        while num_tries < max_tries:
            # Generate random perturbations for tangents
            m_x_perturbed = m_x * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
            m_y_perturbed = m_y * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
            # m_x_perturbed = m_x * np.random.uniform(size=m_x.shape[0]) * rand_scale
            # m_y_perturbed = m_y * np.random.uniform(size=m_x.shape[0]) * rand_scale
            
            # Generate points on the cubic Hermite curve
            points = cubic_hermite(x, y, m_x_perturbed, m_y_perturbed, num_points, lims=lims)
            
            # Check if points satisfy non-negative constraint if enabled
            if not non_negative or (non_negative and not np.any(points < 0)):
                all_points.append(points)
                break
            num_tries += 1
            
        if num_tries == max_tries:
            print(f"Warning: Could not find valid curve after {max_tries} attempts")
    
    # Stack all points into a single array
    all_points_st = np.stack(all_points)
    if non_negative:
        assert np.all(all_points_st >= 0), "points are negative"
    if return_alpha:
        return all_points_st, np.linspace(lims[0], lims[1], num_points)
    return all_points_st