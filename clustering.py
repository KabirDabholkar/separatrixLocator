import torch
from sklearn.cluster import KMeans
import numpy as np


def get_cluster_centroids(data, k=2):
    """
    Perform k-means clustering on a tensor of data using sklearn.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_dims, ..., feature_dim]
        k (int): Number of clusters (default is 2)

    Returns:
        torch.Tensor: The centroids of the clusters, shape [k, feature_dim]
    """
    # Reshape the input tensor to [num_samples, feature_dim] if necessary
    x_flat = data.reshape(-1, data.shape[-1]).cpu().numpy()  # Convert to NumPy array

    # Perform K-means clustering using sklearn
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x_flat)

    # Calculate within-cluster variation (sum of squared distances to the centroid)
    within_variation = np.std((x_flat - kmeans.cluster_centers_[kmeans.labels_]))

    # Calculate the total variation (sum of squared distances to the global mean)
    global_mean = np.mean(x_flat, axis=0,  keepdims=True)
    total_variation = np.std((x_flat - global_mean))
    print(
        'Within cluster std ', within_variation,
        '\nTotal std', total_variation
    )

    # Return the centroids as a torch tensor
    return torch.tensor(kmeans.cluster_centers_)
