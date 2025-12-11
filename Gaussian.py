import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import diags

def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true

def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)

    # ...
    for j in range(n):
        for i in range(j+1):
            U[i,j] = A[i,j] - np.sum(L[i,:i]*U[:i, j])

        for i in range(j+1, n):
            L[i,j] = (A[i, j] - np.sum(L[i,:j] *U[:j,j]))/U[j,j]

        L[j,j] = 1

    return L,U


A = np.array([[4, 2, 0], [2, 3, 1], [0, 1, 2.5]], dtype=float)
L, U = lu_factorisation(A)
print("L:\n", L)
print("U:\n", U)
print("A from LU:\n", L @ U)




def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U



def system_size(A, b):
    """
    Validate the dimensions of a linear system and return its size.

    This function checks whether the given coefficient matrix `A` is square
    and whether its dimensions are compatible with the right-hand side vector
    `b`. If the dimensions are valid, it returns the size of the system.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D array of shape ``(n, n)`` representing the coefficient matrix of
        the linear system.
    b : numpy.ndarray
        A array of shape ``(n, o)`` representing the right-hand side vector.

    Returns
    -------
    int
        The size of the system, i.e., the number of variables `n`.

    Raises
    ------
    ValueError
        If `A` is not square or if the size of `b` does not match the number of
        rows in `A`.
    """

    # Validate that A is a 2D square matrix
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, but got {A.ndim}D array")

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A must be square, but got A.shape={A.shape}")

    if b.shape[0] != n:
        raise ValueError(
            f"System shapes are not compatible: A.shape={A.shape}, "
            f"b.shape={b.shape}"
        )

    return n


def row_swap(A, b, p, q):
    """
    Swap two rows in a linear system of equations in place.

    This function swaps the rows `p` and `q` of the coefficient matrix `A`
    and the right-hand side vector `b` for a linear system of equations
    of the form ``Ax = b``. The operation is performed **in place**, modifying
    the input arrays directly.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the linear system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector of the system.
    p : int
        The index of the first row to swap. Must satisfy ``0 <= p < n``.
    q : int
        The index of the second row to swap. Must satisfy ``0 <= q < n``.

    Returns
    -------
    None
        This function modifies `A` and `b` directly and does not return
        anything.
    """
    # get system size
    n = system_size(A, b)
    # swap rows of A
    for j in range(n):
        A[p, j], A[q, j] = A[q, j], A[p, j]
    # swap rows of b
    b[p, 0], b[q, 0] = b[q, 0], b[p, 0]


def row_scale(A, b, p, k):
    """
    Scale a row of a linear system by a constant factor in place.

    This function multiplies all entries in row `p` of the coefficient matrix
    `A` and the corresponding entry in the right-hand side vector `b` by a
    scalar `k`. The operation is performed **in place**, modifying the input
    arrays directly.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the linear system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector of the system.
    p : int
        The index of the row to scale. Must satisfy ``0 <= p < n``.
    k : float
        The scalar multiplier applied to the entire row.

    Returns
    -------
    None
        This function modifies `A` and `b` directly and does not return
        anything.
    """
    n = system_size(A, b)

    # scale row p of A
    for j in range(n):
        A[p, j] = k * A[p, j]
    # scale row p of b
    b[p, 0] = b[p, 0] * k


def row_add(A, b, p, k, q):
    """
    Perform an in-place row addition operation on a linear system.

    This function applies the elementary row operation:

    ``row_p ← row_p + k * row_q``

    where `row_p` and `row_q` are rows in the coefficient matrix `A` and the
    right-hand side vector `b`. It updates the entries of `A` and `b`
    **in place**, directly modifying the original data.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the linear system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector of the system.
    p : int
        The index of the row to be updated (destination row). Must satisfy
        ``0 <= p < n``.
    k : float
        The scalar multiplier applied to `row_q` before adding it to `row_p`.
    q : int
        The index of the source row to be scaled and added. Must satisfy
        ``0 <= q < n``.
    """
    n = system_size(A, b)

    # Perform the row operation
    for j in range(n):
        A[p, j] = A[p, j] + k * A[q, j]

    # Update the corresponding value in b
    b[p, 0] = b[p, 0] + k * b[q, 0]


def gaussian_elimination(A, b, verbose=False):
    """
    Perform Gaussian elimination to reduce a linear system to upper triangular
    form.

    This function performs **forward elimination** to transform the coefficient
    matrix `A` into an upper triangular matrix, while applying the same
    operations to the right-hand side vector `b`. This is the first step in
    solving a linear system of equations of the form ``Ax = b`` using Gaussian
    elimination.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector.
    verbose : bool, optional
        If ``True``, prints detailed information about each elimination step,
        including the row operations performed and the intermediate forms of
        `A` and `b`. Default is ``False``.

    Returns
    -------
    None
        This function modifies `A` and `b` **in place** and does not return
        anything.
    """
    # find shape of system
    n = system_size(A, b)

    # perform forwards elimination
    for i in range(n - 1):
        # eliminate column i
        if verbose:
            print(f"eliminating column {i}")
        for j in range(i + 1, n):
            # row j
            factor = A[j, i] / A[i, i]
            if verbose:
                print(f"  row {j} |-> row {j} - {factor} * row {i}")
            row_add(A, b, j, -factor, i)

        if verbose:
            print()
            print("new system")
            print(A)
            print(b)
            print()

def forward_substitution(A, b):
    """
    Solve a lower triangular system of linear equations using forward
    substitution.

    This function solves the system of equations:

    .. math::
        A x = b

    where `A` is a **lower triangular matrix** (all elements above the main
    diagonal are zero). The solution vector `x` is computed sequentially by
    solving each equation starting from the first row.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the lower triangular
        coefficient matrix of the system.
    b : numpy.ndarray
        A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
        ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape ``(n,)`` containing the solution vector.
    """
    """
    solves the system of linear equationa Ax = b assuming that A is lower
    triangular. returns the solution x
    """
    # get size of system
    n = system_size(A, b)

    # check is lower triangular
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix A is not lower triangular")

    # create solution variable
    x = np.empty_like(b)

    # perform forwards solve
    for i in range(n):
        partial_sum = 0.0
        for j in range(0, i):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x


def backward_substitution(A, b):
    """
    Solve an upper triangular system of linear equations using backward
    substitution.

    This function solves the system of equations:

    .. math::
        A x = b

    where `A` is an **upper triangular matrix** (all elements below the main
    diagonal are zero). The solution vector `x` is computed starting from the
    last equation and proceeding backward.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the upper triangular
        coefficient matrix of the system.
    b : numpy.ndarray
        A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
        ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape ``(n,)`` containing the solution vector.
    """
    # get size of system
    n = system_size(A, b)

    # check is upper triangular
    assert np.allclose(A, np.triu(A))

    # create solution variable
    x = np.empty_like(b)

    # perform backwards solve
    for i in range(n - 1, -1, -1):  # iterate over rows backwards
        partial_sum = 0.0
        for j in range(i + 1, n):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x


def lu_solve(A, b):
    """
    Solve Ax = b using LU factorisation.

    Parameters
    ----------
    A : ndarray
        Coefficient matrix (n x n)
    b : ndarray
        Right-hand side vector (n,) or (n,1)

    Returns
    -------
    x : ndarray
        Solution vector (n,)
    timings : dict
        Timing of each step: LU, forward, backward
    """
    timings = {}

    # Ensure b is a column vector
    b = b.reshape(-1,1)

    # LU factorisation
    start = time.time()
    L, U = lu_factorisation(A)
    timings['LU'] = time.time() - start

    # Forward substitution
    start = time.time()
    y = forward_substitution(L, b)
    timings['Forward'] = time.time() - start

    # Backward substitution
    start = time.time()
    x = backward_substitution(U, y)
    timings['Backward'] = time.time() - start

    # Flatten solution to 1D
    x = x.flatten()

    return x, timings


# Compare with Gaussian elimination
def gaussian_solve(A, b):
    """
    Solve Ax = b using Gaussian elimination (no pivoting)
    """
    start = time.time()
    A_copy = A.copy()
    b_copy = b.copy()
    gaussian_elimination(A_copy, b_copy)
    x = backward_substitution(A_copy, b_copy)
    return x, time.time() - start


sizes = [2**j for j in range(1, 6)]
lu_times, gauss_times = [], []

for n in sizes:
    A, b, x_true = generate_safe_system(n)
    b = b.reshape(-1,1)

    # LU solver
    _, timings = lu_solve(A, b)
    lu_times.append(sum(timings.values()))

    # Gaussian solver
    _, gauss_time = gaussian_solve(A, b)
    gauss_times.append(gauss_time)

plt.plot(sizes, lu_times, label='LU')
plt.plot(sizes, gauss_times, label='Gaussian')
plt.xlabel("system size n")
plt.ylabel("time")
plt.legend()
plt.grid(True)

# Save plot as an image file
plt.savefig("timing_plot.png")
plt.close()  # closes the plot so the script doesn’t hang