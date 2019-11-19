"""Newton-Raphson scheme"""

import numpy


class DidNotConvergeError(Exception):
    pass


def newton_raphson(
    unknowns,
    residuals,
    compute_residuals,
    jacobian,
    compute_jacobian,
    tol=1e-10,
    iterations=500,
    logger=None
):
    """Solves a (nonlinear) system of N equations `residuals(unknowns) = 0`.

    Parameters
    ----------
    unknowns : numpy.ndarray
        Array of shape (N,) for the unknowns.
        The initial content is used as the initial value of the iteration.
        The final content will contain the result.
    residuals : numpy.ndarray
        Array of shape (N,) for storing the residuals.
        The array is reused to avoid memory allocation.
    compute_residuals : callable
        A method that assembles the residuals vector.
        Takes the residuals array as the first
        and the unknowns array as the second argument.
    jacobian : numpy.ndarray
        Array of shape (N, N) for storing the Jacobian matrix.
        The array is reused to avoid memory allocation.
    compute_jacobian : callable
        A method that assembles the Jacobian matrix.
        Takes the jacobian array as the first
        and the unknowns array as the second argument.
    tol : float, optional
        Iteration converges if the 2-norm of residuals is <= tol.
    iterations : int, optional
        Maximum number of iterations.
    logger : Optional[Logger]
        Logger used to track steps to convergence.

    Raises
    ------
    DidNotConvergeError
        if the Newton-Raphson iteration did not converge
        after the maximum number of allowed iterations.
    """

    for n in range(iterations + 1):
        compute_residuals(residuals, unknowns)

        if numpy.linalg.norm(residuals) <= tol:
            if logger:
                logger.info(f"converged after {n} steps")
            return

        compute_jacobian(jacobian, unknowns)

        unknowns -= numpy.linalg.lstsq(jacobian, residuals, rcond=None)[0]

    raise DidNotConvergeError
