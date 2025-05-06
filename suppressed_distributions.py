import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from scipy.integrate import solve_ivp

# def find_leaky_directions_LDA(mu, sigma, run_and_cluster, num_samples=10000, k=5):
#     """
#     Identify directions in high-D Gaussian that lead to leakage into unwanted basins.
#
#     Args:
#         mu:           Center of isotropic Gaussian (D,)
#         sigma:        Standard deviation of original isotropic Gaussian
#         run_and_cluster: Callable that returns basin labels given input samples of shape (N,D)
#         num_samples:  Number of samples to draw for estimating leaky directions
#         k:            Number of discriminant directions to extract
#
#     Returns:
#         leaky_directions: (k, D) array, orthonormal directions to suppress
#     """
#     D = mu.shape[0]
#
#     # 1. Sample from isotropic Gaussian
#     X = np.random.multivariate_normal(mean=mu, cov=sigma ** 2 * np.eye(D), size=num_samples)
#
#     # 2. Classify samples based on dynamics
#     labels = run_and_cluster(X)  # should return array of shape (num_samples,)
#
#     # 3. Binary labels: 0 for good (2 basins), 1 for bad (other basins)
#     is_good = np.isin(labels, [0, 1])
#     y = np.where(is_good, 0, 1)  # Good = 0, Bad = 1
#
#     # 4. Run LDA to separate good from bad samples
#     lda = LDA(n_components=k)
#     lda.fit(X, y)
#
#     # 5. Extract top-k leaky directions (shape: k x D)
#     leaky_directions = lda.scalings_[:, :k].T  # note: scalings_ is (D, n_components)
#
#     return leaky_directions
#
#
# def construct_suppressed_covariance(sigma, leaky_directions, suppression=0.1):
#     """
#     Construct a new covariance matrix with suppressed variance along leaky directions.
#
#     Args:
#         sigma: scalar, original isotropic stddev
#         leaky_directions: (k, D) orthonormal directions
#         suppression: how much to scale the variance along those directions
#
#     Returns:
#         Sigma_mod: (D, D) anisotropic covariance
#     """
#     D = leaky_directions.shape[1]
#     I = np.eye(D)
#
#     V = leaky_directions
#     # Projector onto leaky subspace
#     P = V.T @ V
#
#     # Suppress variance along P, retain original elsewhere
#     Sigma_mod = sigma ** 2 * (I - P + suppression * P)
#
#     return Sigma_mod

# --- LDA to find leaky direction ---
def find_lda_direction(X, labels, leaky_labels=[0]):
    y = np.where(np.isin(labels, leaky_labels), 0, 1)
    print(y)
    lda = LDA(n_components=1)
    lda.fit(X, y)
    return lda.scalings_[:, 0] / np.linalg.norm(lda.scalings_[:, 0])  # normalized


# --- Construct modified covariance ---
def construct_anisotropic_cov(base_sigma, direction, suppression=0.05):
    I = np.eye(2)
    v = direction.reshape(-1, 1)
    P = v @ v.T
    return base_sigma ** 2 * (I - P + suppression * P)

######## test


if __name__ == '__main__':
    # --- Dynamical system: 3-attractor vector field ---
    def dynamics(t, z):
        x, y = z
        dx = x - x ** 3
        dy = -y + 0.5 * np.sin(3 * x)
        return [dx, dy]


    # --- Simulate trajectories ---
    def simulate_trajectory(x0, T=5.0):
        sol = solve_ivp(dynamics, [0, T], x0, t_eval=[T])
        return sol.y[:, -1]


    # --- Cluster end points into 3 basins ---
    def run_and_cluster(X):
        final_pts = np.array([simulate_trajectory(x0) for x0 in X])
        kmeans = KMeans(n_clusters=3, random_state=0).fit(final_pts)
        return kmeans.labels_


    # --- Run demo ---
    np.random.seed(42)
    mu = np.array([0.2, 0.5])
    sigma = 0.8
    num_samples = 2000

    # Step 1: Sample from isotropic Gaussian
    X = np.random.multivariate_normal(mu, sigma ** 2 * np.eye(2), size=num_samples)

    # Step 2: Run dynamics and label final basin
    labels = run_and_cluster(X)

    # Step 3: Find leaky direction using LDA
    lda_dir = find_lda_direction(X, labels)

    # Step 4: Construct anisotropic covariance
    Sigma_aniso = construct_anisotropic_cov(sigma, lda_dir, suppression=0.02)

    # Step 5: Resample with anisotropic Gaussian
    X_mod = np.random.multivariate_normal(mu, Sigma_aniso, size=num_samples)
    labels_mod = run_and_cluster(X_mod)

    # --- Plot results ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for i, (X_plot, labels_plot, title) in enumerate(zip(
            [X, X_mod],
            [labels, labels_mod],
            ["Original Isotropic", "Anisotropic Suppressed"]
    )):
        axs[i].scatter(X_plot[:, 0], X_plot[:, 1], c=labels_plot, cmap='Set1', s=10, alpha=0.7)
        axs[i].quiver(*mu, *lda_dir, scale=5, color='black', label='LDA dir')
        axs[i].set_title(title)
        axs[i].set_xlim(-2, 2)
        axs[i].set_ylim(-2, 2)
        axs[i].set_aspect('equal')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    