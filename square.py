import numpy as np

from scipy.spatial import ConvexHull


####### THIS APPROACH FAILS #####
def order_square_nd_convex(points, atol=1e-1):
    """
    Orders four points in N-dimensional space to form an approximate square using ConvexHull.
    
    Args:
        points (np.ndarray): A (4, D) array of four N-dimensional points.
        atol (float): Absolute tolerance for checking square properties
        
    Returns:
        np.ndarray: The points ordered as a square.
    """
    if points.shape[0] != 4:
        raise ValueError("Input must be a (4, D) array of four points.")
        
    # Create convex hull
    hull = ConvexHull(points)
    
    # Get vertices in order around the hull
    ordered_points = points[hull.vertices]
    
    # Verify square properties
    d01 = np.linalg.norm(ordered_points[0] - ordered_points[1]) 
    d12 = np.linalg.norm(ordered_points[1] - ordered_points[2])
    d23 = np.linalg.norm(ordered_points[2] - ordered_points[3])
    d30 = np.linalg.norm(ordered_points[3] - ordered_points[0])

    if not (np.isclose(d01, d23, atol=atol) and np.isclose(d12, d30, atol=atol)):
        raise ValueError("The points do not form an approximate square.")
        
    return ordered_points

####### USE THIS APPROACH #####
def order_square_nd(points,atol=1e-1):
    """
    Orders four points in N-dimensional space to form an approximate square.

    Args:
        points (np.ndarray): A (4, D) array of four N-dimensional points.

    Returns:
        np.ndarray: The points ordered as a square.
    """
    if points.shape[0] != 4:
        raise ValueError("Input must be a (4, D) array of four points.")

    # Calculate all pairwise distances
    distances = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1)
    np.fill_diagonal(distances, np.inf)  # Ignore diagonal (self-distance)

    # Start with the first point as the first vertex of the square
    square_order = [0]
    while len(square_order) < 4:
        current_point = square_order[-1]
        # Choose the next point as the closest that isn't already selected
        remaining_points = [i for i in range(4) if i not in square_order]
        next_point = remaining_points[np.argmin(distances[current_point, remaining_points])]
        square_order.append(next_point)

    # Rearrange points in the determined order
    ordered_points = points[square_order]

    # Ensure the points form an approximate square by checking distance consistency
    d01 = np.linalg.norm(ordered_points[0] - ordered_points[1])
    d12 = np.linalg.norm(ordered_points[1] - ordered_points[2])
    d23 = np.linalg.norm(ordered_points[2] - ordered_points[3])
    d30 = np.linalg.norm(ordered_points[3] - ordered_points[0])

    if not (np.isclose(d01, d23, atol=atol) and np.isclose(d12, d30, atol=atol)):
        raise ValueError("The points do not form an approximate square.")

    return ordered_points


if __name__ == "__main__":
    # Example points (approximately forming a square)
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ])
    points = points + np.random.normal(0, 1e-2, size=points.shape)
    ordered_points = order_square_nd(points[np.random.permutation(4)])
    print("Ordered square points:\n", np.round(ordered_points))
    ordered_points = order_square_nd_convex(points[np.random.permutation(4)])
    print("Ordered square points:\n", np.round(ordered_points))