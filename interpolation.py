import numpy as np

def cubic_hermite(x, y, m_x, m_y, num_points=100):
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
    alpha = np.linspace(0, 1, num_points)

    # Cubic Hermite interpolation formula
    points = (2 * alpha ** 3 - 3 * alpha ** 2 + 1)[:, np.newaxis] * x + \
             (-2 * alpha ** 3 + 3 * alpha ** 2)[:, np.newaxis] * y + \
             (alpha ** 3 - 2 * alpha ** 2 + alpha)[:, np.newaxis] * m_x + \
             (alpha ** 3 - alpha ** 2)[:, np.newaxis] * m_y

    return points