"""Script that computes the optimal design for a linear model."""

import numpy as np


def random_points_in_unit_ball(num_points: int, dimensions: int) -> np.ndarray:
    """Generates uniformly distributed random points within in a unit ball.

    Args:
        num_points: The number of points to generate.
        dimensions: The number of dimensions to generate.

    Returns:
        A matrix with 'num_points' rows and 'dimensions' columns.
        Each row is a random point in the unit ball.
    """
    rng = np.random.default_rng(1337)
    points = rng.normal(0, 1, (num_points, dimensions))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points_on_sphere = points / norms
    uniform_radii = rng.uniform(0, 1, size=(num_points, 1))
    adjusted_radii = uniform_radii ** (1 / dimensions)
    points_in_ball: np.ndarray = points_on_sphere * adjusted_radii
    return points_in_ball


def covariance_matrix(points: np.ndarray, distribution: np.ndarray) -> np.ndarray:
    """Computes the covariance matrix of a set of points."""
    if len(points.shape) != 2:
        raise ValueError("points should be a matrix")

    if len(distribution.shape) != 1:
        raise ValueError("points should be a vector")

    if distribution.shape[0] != points.shape[0]:
        raise ValueError("points and distribution have different number if points")

    covariance: np.ndarray = np.transpose(points) @ np.diag(distribution) @ points
    return covariance


def compute_optimal_design(points: np.ndarray, num_iterations: int = 10) -> np.ndarray:
    """Computes the optimal design for estimating a linear model.

    The input to the function is a list of n points in a d-dimensional space.
    The output is a probability distribution over the n points.

    Args:
        points: A list of n points in a d-dimensional space. The points are
            represented as a nxd matrix.
        num_iterations: The number of iterations of the iterative algorithm.

    Returns:
        A vector of length n representing the optimal design for each point.
        The vector represents a discrete probability distribution over the n points.
        The entries of the vector represent the probability distribution over

    Raises:
        ValueError: if points is not a matrix, or if num_iterations is negative.
    """
    if len(points.shape) != 2:
        raise ValueError("points should be a matrix")

    if num_iterations < 0:
        raise ValueError("num_iterations should be a non-negative integer")

    n, d = points.shape

    if n <= 0 or d <= 0:
        return np.array([])

    # Start with the uniform distribution over the points.
    distribution: np.ndarray = np.ones(n) / n

    for t in range(num_iterations):
        covariance = covariance_matrix(points, distribution)
        covariance_inv = np.linalg.inv(covariance)
        best_i = -1
        best_value = -np.inf
        for i in range(n):
            point = points[i]
            value = point @ covariance_inv @ point
            if value > best_value:
                best_value = value
                best_i = i

        print(t, best_i, best_value)

        unit = np.zeros(n)
        unit[best_i] = 1.0
        gamma = (best_value / d - 1) / (best_value - 1)
        distribution = distribution * (1.0 - gamma) + gamma * unit

    return distribution


def main() -> None:
    """Entry point for the script."""
    points = random_points_in_unit_ball(num_points=20, dimensions=2)
    np.set_printoptions(precision=9, floatmode="fixed", suppress=True)
    print(f"points: {points}")

    distribution = compute_optimal_design(points, num_iterations=1000)
    print(distribution)


if __name__ == "__main__":
    main()
